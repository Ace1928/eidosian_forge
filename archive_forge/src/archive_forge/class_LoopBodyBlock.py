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
class LoopBodyBlock:
    """
    Captures the body of a Loops subclass into an FX graph.
    In normal cases there will be a 1:1 mapping between LoopBody and
    LoopBodyBlock, hower in the case of ops.masked() the masked out
    operations will manifest as an extra LoopBodyBlock.
    """

    def __init__(self, body: LoopBody, fn: Callable[..., Any], args: List[Any]):
        self.body = body

        def add_index(expr, category, buf_name=None):
            return tracer.create_proxy('call_module', 'get_index', (self.body.add_index_expr(expr, category, buf_name),), {})

        class CaptureIndexing(V.WrapperHandler):
            self.name = 'CaptureIndexing'

            def load(self, name: str, index: sympy.Expr):
                index = add_index(index, 'reads', name)
                return self._inner.load(name, index)

            def store(self, name, index, value, mode=None):
                index = add_index(index, 'writes', name)
                return self._inner.store(name, index, value, mode)

            def store_reduction(self, name, index, value):
                index = add_index(index, 'writes', name)
                return self._inner.store_reduction(name, index, value)

            def reduction(self, dtype, src_dtype, reduction_type, value):
                result = self._inner.reduction(dtype, src_dtype, reduction_type, value)
                if 'welford' in reduction_type:
                    return tuple((result[i] for i in range(3)))
                return result

            def index_expr(self, index, dtype):
                if isinstance(index, (int, sympy.Integer)):
                    return self._inner.constant(int(index), dtype)
                index = add_index(index, 'other')
                return self._inner.index_expr(index, dtype)

            def bucketize(self, values, offsets_name: str, offsets_size: sympy.Expr, indexing_dtype: torch.dtype, right: bool):
                offsets_size = add_index(offsets_size, 'other')
                return self._inner.bucketize(values, offsets_name, offsets_size, indexing_dtype, right)

            @staticmethod
            def masked(mask_proxy, masked_body: Callable[..., Any], other_proxy):
                """
                Recursively capture the masked out body in another LoopBodyBlock
                """
                subblock: LoopBodyBlock

                def shim(mask, other):
                    return V.ops.masked(mask, subblock, other)
                name = self.body.add_submodule(shim, 'masked_subblock')
                subblock = LoopBodyBlock(self.body, masked_body, [])
                self.body.subblocks[name] = subblock
                return tracer.create_proxy('call_module', name, (mask_proxy, other_proxy), {})

            @staticmethod
            def indirect_indexing(index_proxy, size, check=True):
                """
                Flow data from tensors into indexing formulas.
                Introduce a call_module to update the indexing.
                """
                var = self.body.add_indirect(size)

                def set_indirect(new_var):
                    self.body.replace_indirect(var, V.ops.indirect_indexing(new_var, size, check))
                tracer.create_proxy('call_module', self.body.add_submodule(set_indirect, f'set_{var}'), (index_proxy,), {})
                return var

            @staticmethod
            def output(result):
                tracer.create_proxy('output', 'output', (result,), {})
        tracer = torch.fx.Tracer()
        tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
        proxy_ops = tracer.create_proxy('placeholder', 'ops', (), {})
        from .index_propagation import IndexPropagation
        from .sizevars import SimplifyIndexing
        handler: Any = SimplifyIndexing(CaptureIndexing(proxy_ops), self.body.var_ranges)
        if config.constant_and_index_propagation:
            handler = IndexPropagation(handler)
        with V.set_ops_handler(handler):
            ops.output(fn(*args))
        self.graph = tracer.graph

    def __call__(self):
        graph = self.graph
        submodules = self.body.submodules
        return InterpreterShim(graph, submodules).run(V.get_ops_handler())

    def debug_str(self, name='block'):
        code = torch.fx.GraphModule(self.body.submodules, self.graph).code
        return re.sub(';[^\\n]*', '', code.strip().replace('def forward(', f'def {name}('))