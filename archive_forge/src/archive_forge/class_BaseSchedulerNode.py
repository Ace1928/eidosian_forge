import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
class BaseSchedulerNode:

    def __init__(self, scheduler: 'Scheduler', node: ir.Buffer):
        self.scheduler: Scheduler = scheduler
        self.node: ir.Buffer = node
        self.users: List[NodeUser] = []
        self.inverse_users: List[BaseSchedulerNode] = []
        self.node_users: List[BaseSchedulerNode] = []
        self.set_read_writes(node.get_read_writes())
        self.ancestors: Set[str] = set()
        self.min_order: int
        self.max_order: int
        self.last_usage: Set[str] = set()
        self.written = False

    def __repr__(self):
        return f'{type(self).__name__}(name={self.get_name()!r})'

    def debug_str(self) -> str:
        """Longer form printout for trace logs"""
        name = self.get_name()
        lines = [f'{name}: {type(self).__name__}({type(getattr(self, 'node', None)).__name__})', f'{name}.writes = {pformat(self.read_writes.writes)}', f'{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}', f'{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}', f'{name}.users = {self.users}']
        try:
            lines += [self.debug_str_extra()]
        except Exception:
            log.warning('Ignoring error in debug_str()', exc_info=True)
        return '\n'.join(lines).rstrip()

    def debug_str_extra(self) -> str:
        return ''

    def log_details(self):
        log.info('%s: unmet_dependencies = %s, writes = %s', self, self.unmet_dependencies, self.read_writes.writes)

    def update_mutated_names(self, renames: Dict[str, str]):
        self.set_read_writes(self.read_writes.rename(renames))

    def add_mutation_dep(self, dep):
        self.set_read_writes(self.read_writes.with_read(dep))

    def add_fake_dep(self, dep):
        self.set_read_writes(self.read_writes.with_read(dep))

    def set_users(self, users: List['NodeUser']):
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = use.merge(result[id(use.node)])
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def set_last_usage(self, future_used_buffers: Set[str], mutation_real_name: Dict[str, str]):
        used_buffers = self.used_or_aliased_buffer_names()
        used_buffers = {mutation_real_name.get(k, k) for k in used_buffers}
        self.last_usage = used_buffers - future_used_buffers

    def get_aliases(self):
        return self.node.get_alias_names()

    def get_mutations(self):
        return self.node.get_mutation_names()

    def has_aliasing_or_mutation(self):
        return bool(self.get_aliases() or self.get_mutations())

    def set_read_writes(self, rw: dependencies.ReadWrites):
        self.read_writes: dependencies.ReadWrites = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def op_counts(self):
        return self.read_writes.op_counts

    def used_buffer_names(self) -> Set[str]:
        return {dep.name for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)}

    def used_or_aliased_buffer_names(self) -> Set[str]:
        used_names = set()
        for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes):
            used_names.add(dep.name)
            if V.graph.name_to_buffer.get(dep.name):
                layout = V.graph.name_to_buffer[dep.name].get_layout()
                if isinstance(layout, ir.AliasedLayout):
                    used_names.add(layout.view.data.get_name())
        return used_names

    def prune_deps(self):
        self.unmet_dependencies = {dep for dep in self.unmet_dependencies if dep.name not in self.scheduler.available_buffer_names}

    def prune_weak_deps(self):

        def should_prune(dep):
            return isinstance(dep, WeakDep) and dep.name in V.graph.removed_buffers
        to_remove = {dep for dep in self.read_writes.reads if should_prune(dep)}
        self.set_read_writes(self.read_writes.remove_reads(to_remove))

    def prune_redundant_deps(self, name_to_fused_node):
        """
        Prunes weakdeps intended for mutation ordering
        on an upstream fused node if after fusion there is another dependency
        on the fused upstream node, making the weakdep redundant

        In essence this enforces an ordering on fusions. As fusions occur, weakdeps will
        be incrementally removed, enabling other fusions, ensuring they are fused in order.
        """
        name_to_dep_count: Counter[str] = collections.Counter()
        for dep in self.unmet_dependencies:
            if not isinstance(dep, WeakDep):
                name_to_dep_count[name_to_fused_node[dep.name].get_name()] += 1

        def should_prune(dep):
            if isinstance(dep, WeakDep):
                is_redundant = name_to_dep_count[name_to_fused_node[dep.name].get_name()] > 0
                is_self_dep = name_to_fused_node[dep.name] == self
                return is_redundant or is_self_dep
            else:
                return False
        deps_to_prune = {dep for dep in self.unmet_dependencies if should_prune(dep)}
        if deps_to_prune:
            self.unmet_dependencies = self.unmet_dependencies - deps_to_prune
            self.set_read_writes(self.read_writes.remove_reads(deps_to_prune))

    def get_name(self) -> str:
        return self.node.get_name()

    def get_first_name(self) -> str:
        return self.get_name()

    def get_names(self) -> Set[str]:
        return {self.get_name()}

    def get_nodes(self) -> Sequence['BaseSchedulerNode']:
        return [self]

    def get_device(self):
        return self.node.get_device()

    def is_reduction(self):
        return False

    def is_template(self):
        return False

    def is_extern(self):
        return False

    def is_foreach(self):
        return False

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def has_side_effects(self):
        return False

    def decide_inplace_update(self):
        """
        Decide if there should be inplace updates for the node
        and record the decision in the active kernel.
        """
        if not self.node.should_allocate():
            return
        if isinstance(self, (SchedulerNode,)) and (self.node.get_alias_names() or self.node.get_mutation_names()):
            return
        if (isinstance(self, (SchedulerNode,)) or (isinstance(self, ExternKernelSchedulerNode) and isinstance(self.node, (ir.AllReduce, ir.InPlaceHint)))) and config.inplace_buffers and (not isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel) or getattr(V.kernel, 'mutations', None) is not None):
            from .codegen.wrapper import buffer_reuse_key
            ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)
            for read in ordered_reads:
                input_node: Optional[BaseSchedulerNode] = self.scheduler.name_to_node.get(read.name)
                if input_node and V.graph.wrapper_code.can_reuse(input_node, self):
                    assert input_node.users is not None
                    remaining_uses = [x for x in input_node.users if x.node.get_name() not in self.scheduler.available_buffer_names]
                    if len(remaining_uses) == 1 and remaining_uses[0].can_inplace and (remaining_uses[0].node is self) and (not isinstance(input_node.node.get_layout(), (ir.MultiOutputLayout, ir.MutationLayout, ir.AliasedLayout))) and (not (isinstance(input_node.node, ir.FallbackKernel) and len(input_node.node.get_alias_names()) > 0)) and (buffer_reuse_key(input_node.node) == buffer_reuse_key(self.node)):
                        if hasattr(V.kernel, 'args'):
                            V.kernel.args.make_inplace(input_node.get_name(), self.get_name())
                            if isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel):
                                V.kernel.mutations.add(input_node.get_name())
                                V.kernel.mutations.add(self.get_name())
                            self.last_usage.discard(input_node.get_name())
                            V.kernel.inplace_update_buffers[self.get_name()] = input_node.get_name()
                        break

    def allocate(self):
        if not self.node.should_allocate():
            return
        if isinstance(self, (SchedulerNode,)) and (self.node.get_alias_names() or self.node.get_mutation_names()):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return
        if hasattr(V.kernel, 'args') and self.get_name() in V.kernel.inplace_update_buffers:
            V.graph.wrapper_code.codegen_inplace_reuse(self.scheduler.name_to_node[V.kernel.inplace_update_buffers[self.get_name()]].node, self.node)
        else:
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def codegen_originating_info(self, buffer, only_once=True):
        if not config.comment_origin:
            return
        if only_once and self.written:
            return
        origins = self.node.origins
        out_lines = []
        for o in origins:
            if o.op == 'output':
                continue
            out_lines.append('')
            out_lines.append('#pragma CMT ORIGIN:')
            op_info_str = f'#pragma CMT {o.op} {o.target}'
            if 'seq_nr' in o.meta:
                op_info_str = op_info_str + f' seq_nr:{o.meta['seq_nr']}'
            out_lines.append(op_info_str)
            if 'stack_trace' in o.meta:
                stack_trace = f'{o.meta['stack_trace']}'
                stack_trace_last_line = stack_trace.split('|')[-1]
                out_lines.append('#pragma CMT ' + stack_trace_last_line.replace('{', '{{').replace('}', '}}').replace('\n', '\\'))
                out_lines.append('#pragma CMT END ORIGIN')
                out_lines.append('')
        if len(out_lines) == 0:
            return
        buffer.writelines(out_lines)
        self.written = True

    def get_read_write_buffers_sizes(self) -> int:
        """
        Counting the number of bytes accessed for a kernel is
        surprisingly tricky. In particular, there is a differentiation
        between 'theoretical' memory accesses and practical memory
        accesses. For example, a layernorm kernel may actually access an
        input 3 times, but in theory, it only needs to access its input
        once (and may be optimized to do so through say, persistent
        reductions)

        Another example is that even though a buffer is passed in, we may
        not access the entire buffer. This may occur if we are accessing
        a slice of the buffer. Another tricky case is for indirect
        indexing, where the amount of bytes accessed depends on the
        values of the input.

        What this function aims to compute is the memory accesses for
        worst-case inputs, best-case optimization. What this means is
        that for each buffer we compute the amount of potential accesses in two ways and take the minimum.

        1. Numel in ranges multiplied by number of deps the buffer has
        2. The buffer size
        """
        if isinstance(self, NopKernelSchedulerNode):
            return 0
        if isinstance(self, ExternKernelSchedulerNode) and isinstance(self.node, MultiOutput):
            return 0
        if isinstance(self, SchedulerNode):
            node_numel = V.graph.sizevars.size_hint(sympy_product(self.get_ranges()[0]) * sympy_product(self.get_ranges()[1]))
        else:
            node_numel = int(1000000000.0)
        buf_accesses = collections.defaultdict(list)
        for dep in self.read_writes.reads | self.read_writes.writes:
            buf_accesses[dep.name].append(dep)
        reads = {dep.name for dep in self.read_writes.reads}
        writes = {dep.name for dep in self.read_writes.writes}

        def is_materialized(buf, snodes):
            users = self.scheduler.name_to_node[buf].users
            buf_uses = {user.node for user in users}
            return len(buf_uses - set(snodes)) > 0
        if isinstance(self, FusedSchedulerNode):
            removed_buffers = {dep for dep in writes if not is_materialized(dep, self.snodes)}
            writes = writes - removed_buffers
            reads = reads - removed_buffers
        node_bytes = 0
        for buf_name in reads | writes:
            buf_accessed_elems = sum([node_numel for dep in buf_accesses[buf_name]])
            buf: Union[ir.Buffer, ir.TensorBox]
            if buf_name in V.graph.name_to_buffer:
                buf = V.graph.name_to_buffer[buf_name]
            elif buf_name in V.graph.graph_inputs:
                buf = V.graph.graph_inputs[buf_name]
            else:
                continue

            def get_buf_elems(buf):
                return V.graph.sizevars.size_hint(sympy_product(buf.get_size()))
            if isinstance(buf.layout, MultiOutputLayout):
                users = self.scheduler.name_to_node[buf.get_name()].users
                buf_elems = sum((get_buf_elems(user.node.node) for user in users))
            else:
                buf_elems = get_buf_elems(buf)
            node_bytes += min(buf_elems, buf_accessed_elems) * get_dtype_size(buf.get_dtype())
        return node_bytes

    def get_estimated_runtime(self) -> float:
        """
        Returns estimated op runtime in nanoseconds (ns)
        """
        layout = None
        dtype = None
        if not hasattr(self, 'node') or not self.node:
            assert isinstance(self, (FusedSchedulerNode, ForeachKernelSchedulerNode)), f'type(self)={type(self)!r}'
            assert self.snodes
            if not self.snodes[0].node:
                return 0
            layout = self.snodes[0].node.get_layout()
            dtype = self.snodes[0].node.get_dtype()
        else:
            layout = self.node.get_layout()
            dtype = self.node.get_dtype()
        if 'cuda' != layout.device.type:
            return 0
        try:
            gpu_memory_bandwidth = get_gpu_dram_gbps()
            gpu_flops = get_device_tflops(dtype) * 10 ** 12
        except Exception:
            return 0
        if isinstance(self, ExternKernelSchedulerNode):
            assert isinstance(self.node, ir.ExternKernel), f'type(self.node)={type(self.node)!r}'
            op = kernel_name_to_op.get(getattr(self.node, 'kernel', ''), None)
            if op is not None:
                from torch._subclasses.fake_tensor import FakeTensorMode
                from torch.utils.flop_counter import FlopCounterMode
                with FakeTensorMode(), FlopCounterMode(display=False) as flop_counter_mode:
                    from .ir import ir_node_to_tensor
                    fake_inputs = [ir_node_to_tensor(input, guard_shape=False) for input in self.node.inputs]
                    cls = self.node.__class__
                    cls.process_kernel(op, *fake_inputs, **self.node.kwargs)
                    factor = 1.0
                    counted_flops = flop_counter_mode.get_total_flops()
                    counted_bytes = self.get_read_write_buffers_sizes()
                    compute_time = factor * counted_flops / gpu_flops * 1000000000.0
                    transfer_time = counted_bytes / gpu_memory_bandwidth
                    return max(compute_time, transfer_time)
        elif isinstance(self, FusedSchedulerNode) or isinstance(self.node, ComputedBuffer):
            return self.get_read_write_buffers_sizes() / gpu_memory_bandwidth
        if isinstance(self.node, ir.CollectiveKernel):
            return estimate_nccl_collective_runtime(self)
        elif isinstance(self.node, ir.Wait):
            return 0
        return 0