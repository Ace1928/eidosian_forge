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
class IRNode:
    _current_origins: ClassVar[Set[Any]] = set()

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: Set[torch.fx.Node]):
        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    def __post_init__(self):
        self.origins = set(self._current_origins)
        self.traceback = traceback.format_stack() if config.debug_ir_traceback else None

    def get_traceback(self):
        return self.traceback

    def common_repr(self):
        origins = f'origins={getattr(self, 'origins', '')}'
        if len(origins) > 64:
            origins = f'{origins[:61]}...'
        return [origins]

    def str_helper(self, lines):
        lines = lines + self.common_repr()
        lines = indent(',\n'.join(map(str, lines)))
        return f'{type(self).__name__}(\n{lines}\n)'

    def is_user_of(self, name):
        return name in self.get_read_names()

    @cache_on_self
    def get_read_names(self):
        return {dep.name for dep in self.get_reads()}

    def get_layout(self):
        raise NotImplementedError(f'get_layout() is not implemented by {type(self)}!')

    def get_size(self):
        raise NotImplementedError(f'get_size() is not implemented by {type(self)}!')

    def get_numel(self):
        return sympy_product(self.get_size())

    def is_zero_elements(self):
        return V.graph.sizevars.is_expr_static_and_true(sympy.Eq(self.get_numel(), 0))

    def realize(self):
        """
        If the IRNode refers to data which has not been materialized (e.g.,
        it is a Pointwise/Reduction that could potentially have more
        compute fused into it), realize the IRNode into physical memory,
        ending the possibility of fusing into it, but allowing, e.g., multiple
        users to access the data without having to recompute.

        Check StorageBox.realize for a particularly notable implementation.

        TODO(ezyang): I think, in principle, every IRNode should have an
        implementation of this, and most of the time no-op is OK, but you
        really do have to audit each IRNode for this, so for now, raise
        an error if it's not implemented.  Note that some code in graph.py
        will catch this thrown error and suppress it with a warning.
        """
        raise NotImplementedError(f'realize NYI on {type(self)}')

    def codegen_reference(self, writer=None):
        raise NotImplementedError(f'codegen_reference NYI on {type(self)}')
    get_device: Callable[[], torch.device]
    get_dtype: Callable[[], torch.dtype]
    get_name: Callable[[], str]
    get_reads: Callable[[], Any]
    get_stride: Callable[[], Any]
    get_storage_numel: Callable[[], Any]
    has_exceeded_max_reads: Callable[[], bool]
    make_loader: Callable[[], Callable[[Any], Any]]
    make_indexer: Callable[[], Callable[[Any], Any]]
    mark_reuse: Callable[[int], None]
    realize_hint: Callable[[], None]