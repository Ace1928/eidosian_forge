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
class SliceView(View):

    @classmethod
    def create(cls, x, dim, start, end, step=1):
        step = sympy.expand(step)
        assert step > 0
        try:
            if start == 0 and end >= 2 ** 63 - 1 and (step == 1):
                return x
        except TypeError:
            pass
        sizevars = V.graph.sizevars
        new_size = list(x.get_size())
        start = cls.handle_negative_index(start, new_size[dim])
        end = cls.handle_negative_index(end, new_size[dim])
        if free_unbacked_symbols(start) or free_unbacked_symbols(end):
            end = sympy.Min(end, new_size[dim])
            start = sympy.Min(start, end)
        else:
            end = sizevars.evaluate_min(end, new_size[dim])
            start = sizevars.evaluate_min(start, end)
        new_size[dim] = FloorDiv(end - start + (step - 1), step)
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(old_layout.device, old_layout.dtype, new_size, new_stride, old_layout.offset + old_layout.stride[dim] * start)
            return ReinterpretView(storage, new_layout)

        def reindex(index):
            assert len(index) == len(new_size), f'wrong ndim {index} {new_size}'
            index = list(index)
            index[dim] = index[dim] * step + start
            return index
        return SliceView(x, size=new_size, reindex=reindex)