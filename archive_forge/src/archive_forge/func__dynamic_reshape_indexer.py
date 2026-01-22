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
@staticmethod
def _dynamic_reshape_indexer(old_size, new_size):
    """
        Perform a reshape entirely by modifying indexing math
        """
    size_hint = V.graph.sizevars.size_hint
    vars = [sympy_symbol(f'view{i}') for i in range(len(new_size))]
    stack_new = list(zip(vars, new_size))
    stack_old = list(old_size)
    view_expr = []
    while stack_new and stack_old:
        size_old = stack_old.pop()
        var, size_new = stack_new.pop()
        if size_old == 1:
            view_expr.append(sympy.Integer(0))
            stack_new.append((var, size_new))
        elif size_new == 1:
            stack_old.append(size_old)
        elif size_hint(size_new) == size_hint(size_old):
            view_expr.append(var)
            V.graph.sizevars.guard_equals(size_new, size_old)
        elif size_hint(size_new) < size_hint(size_old):
            while size_hint(size_new) < size_hint(size_old):
                var2, size_new2 = stack_new.pop()
                var = var2 * size_new + var
                size_new = size_new * size_new2
            view_expr.append(var)
            V.graph.sizevars.guard_equals(size_new, size_old)
        elif size_hint(size_new) > size_hint(size_old):
            divisor = sympy.Integer(1)
            modulus = size_old
            view_expr.append(ModularIndexing(var, divisor, modulus))
            divisor = divisor * modulus
            while size_hint(size_new) > size_hint(size_old):
                modulus = stack_old.pop()
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                size_old = size_old * modulus
            V.graph.sizevars.guard_equals(size_new, size_old)
        else:
            raise AssertionError()
    while stack_old:
        size_old = stack_old.pop()
        V.graph.sizevars.guard_equals(size_old, 1)
        view_expr.append(sympy.Integer(0))
    while stack_new:
        var, size_new = stack_new.pop()
        V.graph.sizevars.guard_equals(size_new, 1)
    view_expr = list(reversed(view_expr))
    assert len(view_expr) == len(old_size)

    def reindex(index):
        assert len(index) == len(vars), (len(index), len(vars))
        replacements = dict(zip(vars, index))
        return tuple((sympy_subs(x, replacements) for x in view_expr))
    return reindex