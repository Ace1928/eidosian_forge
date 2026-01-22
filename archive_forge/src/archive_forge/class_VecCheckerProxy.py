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
class VecCheckerProxy:
    bin_cmp_ops = ['eq', 'ne', 'le', 'ge', 'lt', 'gt']

    @staticmethod
    def _bin_cmp_op(x, y):
        current_node: torch.fx.Node = V.interpreter.current_node
        if not self.is_supported_cmp(current_node):
            self.disable_vec(f'binary comparison op: {current_node}')
        return self.simd_vec

    @staticmethod
    def __getattr__(name):

        def inner(*args, **kwargs):
            if name in VecCheckerProxy.bin_cmp_ops:
                return VecCheckerProxy._bin_cmp_op(args, kwargs)
            if name not in self.fast_vec_list:
                self.disable_vec(f'op: {name}')
            return self.simd_vec
        return inner

    @staticmethod
    def load(name: str, index: sympy.Expr):
        return self.load(name, index)

    @staticmethod
    def store(name, index, value, mode=None):
        return self.store(name, index, value, mode=mode)

    @staticmethod
    def reduction(dtype, src_dtype, reduction_type, value):
        return self.reduction(dtype, src_dtype, reduction_type, value)

    @staticmethod
    def store_reduction(name, index, value):
        return self.store_reduction(name, index, value)

    @staticmethod
    def constant(val, dtype):
        with RecordOptimizationContext(__name__) as node_ctx:
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            i32_iinfo = torch.iinfo(torch.int32)
            if dtype == torch.int64 and val <= i32_iinfo.max and (val >= i32_iinfo.min):
                opt_ctx.dtype = torch.int32
            f32_iinfo = torch.finfo(torch.float32)
            if dtype == torch.double:
                if val <= f32_iinfo.max and val >= f32_iinfo.min or val == torch.inf or val == -torch.inf:
                    opt_ctx.dtype = torch.float32
            supported_dtypes = [torch.float32, torch.int32, torch.bfloat16, torch.float16]
            if opt_ctx.dtype not in supported_dtypes or (opt_ctx.dtype == torch.int32 and (not all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)))):
                self.disable_vec(f'constant dtype: {opt_ctx.dtype}')
            return val

    @staticmethod
    def index_expr(expr, dtype):
        assert len(self.ranges) == len(self.itervars)
        if not len(self.ranges) or not all((not isinstance(range, sympy.Expr) or sympy.simplify(range).is_number for range in self.ranges)):
            self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
            return self.cse.newvar()

        def can_use_int32():
            free_symbols = list(expr.free_symbols)
            sizes = {k: v for k, v in zip(self.itervars, self.ranges) if k in free_symbols}
            if any((v == 0 for v in sizes.values())):
                return True
            vars_ranges = {k: ValueRanges(0, v - 1) for k, v in sizes.items()}
            if not vars_ranges or len(vars_ranges) != len(free_symbols):
                i32_iinfo = torch.iinfo(torch.int32)
                return expr.is_number and expr <= i32_iinfo.max and (expr >= i32_iinfo.min)
            expr_ranges = bound_sympy(expr, vars_ranges)
            if math.isinf(expr_ranges.lower) or math.isinf(expr_ranges.upper):
                return False
            return range_expressable_in_32_bits(ValueRanges(int(expr_ranges.lower), int(expr_ranges.upper) + 1))
        with RecordOptimizationContext(__name__) as node_ctx:
            assert len(self.ranges) == len(self.itervars)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            if dtype == torch.int64 and can_use_int32() and all((user.target in VecCheckerProxy.bin_cmp_ops for user in node_ctx.current_node.users)):
                opt_ctx.dtype = torch.int32
            else:
                opt_ctx.dtype = dtype
                self.disable_vec(f'index_expr: {expr}, dtype {dtype}')
            tiling_var = self.itervars[self.tiling_idx]
            tiling_var_irrelevant = not expr.has(tiling_var)
            if not tiling_var_irrelevant:
                self.disable_vec(f'index_expr (tiling var relevant): {expr}, dtype {dtype}')
            opt_ctx.is_most_inner_loop_irrevelant = tiling_var_irrelevant
            tmp_var = self.cse.newvar()
            return tmp_var

    @staticmethod
    def indirect_indexing(index_var, size, check=True):
        return sympy_symbol(str(index_var))

    @staticmethod
    def masked(mask, body, other):
        body()
        return self.cse.newvar()

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        with RecordOptimizationContext(__name__) as node_ctx:
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = dtype
            cur_node = node_ctx.get_fx_node()
            input_value: torch.fx.Node = cur_node.all_input_nodes[1]
            if dtype == torch.float:
                if input_value.target in ['load']:
                    dtype = V.graph.get_dtype(input_value.args[1]) if input_value.target == 'load' else input_value.args[-1]
                    if dtype in [torch.float16, torch.bfloat16, torch.float, torch.uint8]:
                        pass
                    elif dtype in [torch.int32, torch.int64] and input_value.target == 'load':
                        buffer = V.graph.get_buffer(input_value.args[1])
                        if not (isinstance(buffer, TensorBox) and isinstance(buffer.data, StorageBox) and (len(buffer.data.layout.size) == 0)):
                            self.disable_vec(f'to_dtype: dtype {dtype}')
                    else:
                        self.disable_vec(f'to_dtype: dtype {dtype}')
            elif dtype in DTYPE_LOWP_FP:
                if not all((usr.target == 'store' for usr in cur_node.users)):
                    self.disable_vec('to_dtype: bfloat16/float16 expecting users are all stores')
                    return x
                store_names = [usr.args[1] for usr in cur_node.users]
                if not all((V.graph.get_dtype(name) in [dtype] for name in store_names)):
                    self.disable_vec('to_dtype: expecting all stores into bfloat16 or float16')
                    return x
            elif dtype == torch.bool:
                pass
            elif dtype == torch.uint8:
                is_to_uint8_and_store = all((usr.target in ['store'] for usr in cur_node.users))
                is_to_uint8_and_to_float = all((usr.target in ['to_dtype'] and usr.args[2] == torch.float32 for usr in cur_node.users))
                if not (is_to_uint8_and_store or is_to_uint8_and_to_float):
                    self.disable_vec(f'to_dtype: dtype {dtype}')
            else:
                self.disable_vec(f'to_dtype: dtype {dtype}')
            return x