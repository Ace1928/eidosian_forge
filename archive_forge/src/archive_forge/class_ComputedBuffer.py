import collections
import contextlib
import dataclasses
import functools
import itertools
import logging
import re
import textwrap
import traceback
from contextlib import nullcontext
from enum import Enum
from functools import partial
from inspect import signature
from typing import (
from unittest.mock import patch
import sympy
from sympy import Expr, Integer
import torch._export.serde.schema as export_schema
import torch._logging
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.device_interface import get_interface_for_device
from torch._dynamo.utils import identity
from torch._export.serde.serialize import GraphModuleSerializer
from torch._prims_common import (
from torch._subclasses.fake_tensor import get_schema_info
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols, SymTypes
from torch.utils._sympy.functions import CleanDiv, FloorDiv, ModularIndexing
from . import config, dependencies
from .codegen.common import index_prevent_reordering
from .dependencies import (
from .utils import (
from .virtualized import ops, V
@dataclasses.dataclass
class ComputedBuffer(Buffer):
    data: Loops

    def get_computed_buffer_name(self):
        """
        Returns self.name if it exists, otherwise returns the name of the data node if that exists.
        If neither exist, returns None.
        """
        if self.name is not None:
            return self.name
        if hasattr(self.data, 'name'):
            return self.data.name
        return None

    @cache_on_self
    def num_reads(self):
        return len(self.get_read_writes().reads)

    def get_read_writes(self):
        with patch.object(FlexibleLayout, 'allow_indexing', True):
            if self.data.get_reduction_type():
                return extract_read_writes(self.get_store_function(), self.data.get_size(), self.data.get_reduction_size())
            else:
                return extract_read_writes(self.get_store_function(), self.data.get_size())

    def get_unbacked_symbol_uses(self):
        return free_unbacked_symbols(self.get_size()) | free_unbacked_symbols(self.get_stride()) | free_unbacked_symbols(self.get_offset())

    def make_loader(self):
        if hasattr(self.data, 'make_loader') and self.name not in V.graph.mutated_buffers and (self.num_reads() == 0):
            return self.data.make_loader()
        return super().make_loader()

    def get_store_function(self):
        indexer = self.layout.as_fixed().make_indexer()
        if isinstance(self.data, Reduction):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            assert isinstance(self.data, Pointwise)
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self):
        """
        If our layout is still flexible, try to determine the stride order based on stride orders of reads.

        TODO(jansel): A better algorithm here would look at downstream consumers of this
                      value and try to do global graph-level layout optimization.
                      This is also something just begging to be autotuned.
        """
        if isinstance(self.layout, FlexibleLayout):
            (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(self.data.get_size(), self.data.get_reduction_size())
            reads = self.get_read_writes().reads
            reads_bufs = [V.graph.name_to_buffer[r.name] if r.name in V.graph.name_to_buffer.keys() else None for r in reads]
            assert all((isinstance(r, (dependencies.StarDep, dependencies.MemoryDep)) for r in reads))
            reads = [sympy_subs(r.index, {v: sympy.Integer(0) for v in reduction_vars if v != 0}) for r in reads if isinstance(r, dependencies.MemoryDep)]
            if reads:
                stride_lengths = [V.graph.sizevars.stride_hints(expr, index_vars) for expr in reads]
                from .scheduler import pick_loop_order
                return pick_loop_order(stride_lengths, self.get_size())
        return None

    def decide_layout(self):
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    def simplify_and_reorder(self):
        """
        This is a main place where we do loop transformations in a
        backend-agnostic way.

        Here we:
            1) Remove any 1 dimensions
            2) Fuse contiguous dimensions together
            3) Reorder dimensions based on stride orders
        """
        args, var_ranges = dependencies.index_vars_squeeze(self.data.get_size(), self.data.get_reduction_size(), prefix='q')
        with patch.object(ConstantBuffer, 'override_device', self.get_device()):
            body = LoopBody(self.get_store_function(), args if self.get_reduction_type() else args[:1], var_ranges)
        index_formulas = [*body.indexing_exprs.values()]
        reads_bufs = [V.graph.name_to_buffer[reads_name] if reads_name in V.graph.name_to_buffer.keys() else None for reads_name in body.reads_name2expr.keys()]
        memory_addrs = [*body.reads_name2expr.values(), *body.writes_name2expr.values()]
        index_vars = []
        reduce_vars: List[Any] = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                assert not reduce_vars
                index_vars.append(v)
                index_size.append(s)
            else:
                assert v in args[1]
                reduce_vars.append(v)
                reduce_size.append(s)
        reordering_reindex = [same_reorder(range(len(index_vars)))] * len(memory_addrs)
        for i, reads_buf in enumerate(reads_bufs):
            if isinstance(reads_buf, ComputedBuffer) and hasattr(reads_buf, 'iter_reordering_reindex'):
                reordering_reindex[i] = reads_buf.iter_reordering_reindex

        def simplify_and_reorder(x_vars, support_vars, sizes, reordering_reindex=None):
            sizes, reindex0, reindex1 = self._apply_loop_reordering(x_vars, support_vars, sizes, memory_addrs, reordering_reindex)
            x_vars = reindex0(x_vars)
            sizes, reindex2, prune = V.graph.sizevars._simplify_loops(x_vars, sizes, index_prevent_reordering(index_formulas, x_vars, sizes))
            x_vars = prune(x_vars)
            reindex = fuse_reindexing(reindex1, reindex2)
            return (sizes, reindex, reindex1)
        support_vars = index_vars + reduce_vars
        iter_ranges, iter_reindex, iter_reordering_reindex = simplify_and_reorder(index_vars, support_vars, index_size, reordering_reindex)
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(reduce_vars, support_vars, reduce_size)
        if len(iter_ranges) == len(index_vars):
            self.iter_reordering_reindex = iter_reordering_reindex
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(iter_ranges, reduce_ranges, prefix='z')
        body = LoopBody(body, [iter_reindex(iter_vars), reduce_reindex(reduce_vars)], var_ranges)
        return ((iter_ranges, reduce_ranges), body)

    @staticmethod
    def _apply_loop_reordering(index_vars, support_vars, sizes, memory_addrs, reordering_reindex=None, priority_idx=None):
        """
        Shuffle the order of loops around to hopefully improve performance.
        """
        from .scheduler import pick_loop_order
        if priority_idx is None:
            priority_idx = []
        try:
            strides = [V.graph.sizevars.stride_hints(expr, index_vars, support_vars) for expr in memory_addrs]
            assert len(strides) == len(memory_addrs) and len(strides[0]) == len(index_vars)
            if reordering_reindex is not None:
                for i in range(len(memory_addrs)):
                    try:
                        strides[i] = reordering_reindex[i](strides[i])
                    except AssertionError:
                        pass
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning('Did not simplify complex index:\n%s\n%s', dict(zip(index_vars, sizes)), memory_addrs)
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return (sizes, same_reorder(order), inverse_reorder(order))

    def get_reduction_size(self):
        return self.data.get_reduction_size()

    def get_reduction_type(self):
        return self.data.get_reduction_type()

    def is_no_op(self):
        return self.data.is_zero_elements()

    def should_allocate(self):
        return True

    def constant_to_device(self, device):
        """Move this to a given device. Requires that all reads are to constants."""
        return self.data.constant_to_device(device)