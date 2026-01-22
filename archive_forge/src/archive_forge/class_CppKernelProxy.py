import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppKernelProxy(CppKernel):

    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def data_type_propagation(self, nodes):
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            DataTypePropagation.propagate_scheduler_node(_node)

    def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
        if not isinstance(scheduler_node._body, ir.LoopBody):
            return True
        _lowp_fp_type: Optional[torch.dtype] = None
        DataTypePropagation.propagate_scheduler_node(scheduler_node)
        sub_blocks = [scheduler_node._body.root_block] + list(scheduler_node._body.subblocks.values())
        for sub_block in sub_blocks:
            for _node in sub_block.graph.nodes:
                if _node.op == 'placeholder' or _node.target in ('get_index', 'index_expr'):
                    continue
                if _node.target not in ['load', 'store', 'abs', 'neg', 'output']:
                    return False
                if hasattr(_node, 'meta') and _node.meta:
                    assert OptimizationContext.key in _node.meta
                    opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                    if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                        return False
                    if _lowp_fp_type:
                        assert _lowp_fp_type == opt_ctx.dtype, 'scheduler node do not support bf16/fp16 mix'
                    else:
                        _lowp_fp_type = opt_ctx.dtype
                else:
                    return False
        scheduler_node._lowp_fp_type = _lowp_fp_type
        return True

    def legalize_lowp_fp_dtype(self, nodes):

        def add_to_dtype(sub_graph: torch.fx.Graph):

            def is_lowp_fp_load(node: torch.fx.Node):
                if node.target not in ['load']:
                    return False
                assert len(node.args) == 3
                load_dtype = V.graph.get_dtype(node.args[1])
                return load_dtype in DTYPE_LOWP_FP

            def is_lowp_fp_store(node: torch.fx.Node):
                if node.target != 'store':
                    return False
                _, store_var, _, _, _ = node.args
                store_dtype = V.graph.get_dtype(store_var)
                return store_dtype in DTYPE_LOWP_FP
            sub_graph_nodes = list(sub_graph.nodes)
            to_lowp_fp_legalized_nodes = []
            for _node in sub_graph_nodes:
                if is_lowp_fp_load(_node):
                    if all((user.target == 'store' for user in _node.users)):
                        continue
                    ops = _node.args[0]
                    with sub_graph.inserting_after(_node):
                        to_type_node = sub_graph.call_method('to_dtype', args=(ops, _node, torch.float))
                        to_type_node_args = to_type_node.args
                        _node.replace_all_uses_with(to_type_node)
                        to_type_node.args = to_type_node_args
                        metrics.cpp_to_dtype_count += 1
                elif is_lowp_fp_store(_node):
                    ops, name, _, value_var, _ = _node.args
                    if value_var.target == 'load' and all((user.target == 'store' for user in value_var.users)):
                        continue
                    dtype = V.graph.get_dtype(name)
                    with sub_graph.inserting_before(_node):
                        to_type_node = sub_graph.call_method('to_dtype', args=(ops, value_var, dtype))
                        _node.replace_input_with(value_var, to_type_node)
                        metrics.cpp_to_dtype_count += 1
                elif _node.target == 'reduction':
                    ops, dtype, src_dtype, reduction_type, value = _node.args
                    if src_dtype in DTYPE_LOWP_FP:
                        assert dtype in [torch.float, torch.bfloat16, torch.float16, torch.int64]
                        _node.args = (ops, torch.float if dtype in DTYPE_LOWP_FP else dtype, torch.float, reduction_type, value)
                elif _node.target == 'to_dtype' and _node.args[-1] in DTYPE_LOWP_FP:
                    ops, x, _ = _node.args
                    to_lowp_fp_legalized_nodes.append(_node)
                    _node.args = (ops, x, torch.float)
                else:
                    pass

            def eliminate_to_dtype(sub_graph: torch.fx.Graph):

                def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):

                    def _used_by_to(to_node: torch.fx.Node):
                        return all((usr.target == 'to_dtype' for usr in to_node.users))
                    all_to_nodes = [node for node in sub_graph.nodes if node.target == 'to_dtype']
                    all_to_nodes_and_users = [{node: node.users} for node in all_to_nodes if _used_by_to(node)]
                    for node_users in all_to_nodes_and_users:
                        for node, users in node_users.items():
                            if node in sub_graph.nodes and (all((usr.args[-1] == node.args[-1] for usr in users)) or (node in to_lowp_fp_legalized_nodes and all((usr.args[-1] in DTYPE_LOWP_FP for usr in users)))):
                                val_node = node.all_input_nodes[-1]
                                node.replace_all_uses_with(val_node)
                                sub_graph.erase_node(node)
                    if sub_graph.owning_module is None:
                        sub_graph.lint()
                _eliminate_duplicate_to_node(sub_graph)
            eliminate_to_dtype(sub_graph)

        def _legalize_lowp_fp(loop_body: ir.LoopBody):
            sub_blocks = [loop_body.root_block] + list(loop_body.subblocks.values())
            for sub_block in sub_blocks:
                add_to_dtype(sub_block.graph)
        if all((isinstance(_node, SchedulerNode) and self.is_lowp_fp_scheduler(_node) for _node in nodes)):
            for _node in nodes:
                sub_blocks = [_node._body.root_block] + list(_node._body.subblocks.values())
                for sub_block in sub_blocks:
                    for fx_node in sub_block.graph.nodes:
                        if fx_node.target in ['load', 'store']:
                            assert fx_node.meta
                            assert OptimizationContext.key in fx_node.meta
                            opt_ctx: OptimizationContext = fx_node.meta[OptimizationContext.key]
                            assert opt_ctx.dtype in DTYPE_LOWP_FP
            return
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            assert isinstance(_node._body, ir.LoopBody)
            node: SchedulerNode = _node

            def is_memory_copy_scheduler_node(node: SchedulerNode):
                op_counts = node.read_writes.op_counts
                return len(op_counts) == 2 and 'load' in op_counts and ('store' in op_counts)
            should_legalize = not is_memory_copy_scheduler_node(node)
            if should_legalize:
                body: ir.LoopBody = node._body
                _legalize_lowp_fp(body)

    def codegen_nodes(self, nodes):
        self.legalize_lowp_fp_dtype(nodes)
        self.data_type_propagation(nodes)
        assert len(nodes) >= 1
        first_node = nodes[0]
        vec_dtype = first_node._lowp_fp_type if all((hasattr(_node, '_lowp_fp_type') and _node._lowp_fp_type == first_node._lowp_fp_type for _node in nodes)) else torch.float
        kernel_group = self.kernel_group
        _, (group, reduction_group) = max(nodes, key=lambda x: int(x.is_reduction())).group
        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                run(kernel)
                metrics.generated_kernel_count -= 1
                return kernel

        def run(kernel):
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            in_suffix = False
            for node in nodes:
                if node.group[1] in [(group, reduction_group), (group + reduction_group, ())]:
                    assert not in_suffix
                    node.run(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert node.group[1] == (group, ()), f'unexpected group: {node.group[1]} != {group}, {reduction_group}'
                    with kernel.write_to_suffix():
                        node.run(vars, ())
        scalar_kernel = codegen_kernel(CppKernel)
        V.graph.removed_buffers |= scalar_kernel.removed_buffers
        V.graph.inplaced_to_remove |= scalar_kernel.inplaced_to_remove
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)
        if not self.picked_vec_isa:
            return

        def select_tiling_indices():
            all_index = []
            for node in nodes:
                rw = dependencies.extract_read_writes(node._body, *node._sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
            for index in all_index:
                for var in index.free_symbols:
                    if not re.search('^d\\d+$', var.name):
                        continue
                    stride = stride_at(var, index)
                    if stride == 1:
                        contig_vars.add(int(var.name[1:]))
                        contig_vars_list.append(int(var.name[1:]))
                    elif all((s.name.startswith('s') for s in stride.free_symbols)):
                        non_contig_stride_const.add(int(var.name[1:]))
                    else:
                        non_contig_stride_other.add(int(var.name[1:]))
            contig_only = contig_vars - non_contig_stride_const - non_contig_stride_other
            if len(contig_vars) == 0:
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (contig_vars & non_contig_stride_const) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if len(contig_vars_sorted) == 2 and contig_vars_sorted[-1] in contig_and_const_stride and (contig_vars_sorted[-1] == len(self.itervars) - 1):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]

        def select_tiling(dtype: torch.dtype=torch.float):
            tiling_factor = self.picked_vec_isa.nelements(dtype=dtype)
            tiling_indices = select_tiling_indices()
            if tiling_indices:
                could_vec = True
                for tiling_indice in tiling_indices:
                    with CppVecKernelChecker(deepcopy(self.kernel_group.args), parallel_num_threads(), tiling_factor, tiling_indice) as vec_checker:
                        run(vec_checker)
                        could_vec = could_vec and vec_checker.simd_vec
                        if not could_vec:
                            break
                if could_vec:
                    if len(tiling_indices) == 1:
                        return ([tiling_factor], tiling_indices)
                    if len(tiling_indices) == 2:
                        return ([tiling_factor, tiling_factor], tiling_indices)
            return ([], [])
        with torch._inductor.config.patch(inplace_buffers=False):
            tiling_factors, tiling_indices = select_tiling(vec_dtype)
            assert len(tiling_factors) == len(tiling_indices)
            if len(tiling_indices) == 1:
                main_loop, tail_loop = self.loop_nest.split_with_tiling(tiling_indices[0], factor=tiling_factors[0])
                main_loop.set_kernel(codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype))
                tail_loop.set_kernel(scalar_kernel)
                main_loop.simd_vec = True
                tail_loop.simd_omp = True
                tail_loop.simd_nelements = tiling_factors[0] // 2
            elif len(tiling_indices) == 2:
                assert tiling_indices[1] == len(self.itervars) - 1 and tiling_factors[0] == tiling_factors[1]
                outer_main_loop, outer_tail_loop = self.loop_nest.split_with_tiling(tiling_indices[0], factor=tiling_factors[0])
                outer_tail_loop.set_kernel(scalar_kernel)
                inner_main_loop, inner_tail_loop = outer_main_loop.split_with_tiling(tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0])
                inner_main_loop.set_kernel(codegen_kernel(CppTile2DKernel, tiling_factors[0], tiling_indices, vec_dtype))
                inner_tail_loop.set_kernel(codegen_kernel(CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype))

    def codegen_loops(self, code, worksharing):
        self.codegen_loops_impl(self.loop_nest, code, worksharing)