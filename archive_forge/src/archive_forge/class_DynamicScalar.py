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
class DynamicScalar(ExternKernel):
    """
    The result of a call to aten._local_scalar_dense.
    """

    def get_reads(self):
        return ()

    def should_allocate(self):
        return False

    def __init__(self, sym, data):
        super().__init__(None, NoneLayout(torch.device('cpu')), [data])
        if isinstance(sym, sympy.Symbol):
            self.sym = sym
            self.is_bool = False
        else:
            assert isinstance(sym, sympy.Eq), sym
            assert isinstance(sym.args[0], sympy.Symbol), sym
            assert sym.args[1] == 1, sym
            self.sym = sym.args[0]
            self.is_bool = True

    def get_unbacked_symbol_defs(self):
        return {self.sym}

    def codegen(self, wrapper):
        data, = (t.codegen_reference() for t in self.inputs)
        if self.is_bool:
            wrapper.writeline(f'{self.sym} = 1 if {data}.item() else 0')
        else:
            wrapper.writeline(f'{self.sym} = {data}.item()')
        wrapper.writeline(f'{self.get_name()} = None')