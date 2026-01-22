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
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """
    data: IRNode

    def __getattr__(self, name):
        fn = getattr(self.data, name)
        if callable(fn):
            return fn
        raise AttributeError(f'{type(self.data).__name__}.{name} not callable')

    def realize(self):
        return self.data.realize()

    def codegen_reference(self, writer=None):
        return self.data.codegen_reference(writer)

    @property
    def layout(self):
        return self.data.layout

    def get_layout(self):
        return self.layout

    def get_size(self):
        return self.data.get_size()

    def __str__(self):
        if isinstance(self.data, MutableBox):
            line0 = f'{type(self).__name__}({type(self.data).__name__}('
            endl = '))'
            inner = self.data.data
        else:
            line0 = f'{type(self).__name__}('
            inner = self.data
            endl = ')'
        lines = [line0, indent(str(inner)), endl]
        return '\n'.join(lines)
    __repr__ = __str__