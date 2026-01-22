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
@classmethod
def _multilayer_wrap_loader(cls, loader, reduction_ranges, reduction_numel, split, block_size, default):
    reindex = View.dynamic_reshape_indexer(reduction_ranges, [reduction_numel])
    need_mask = not V.graph.sizevars.is_expr_static_and_true(sympy.Eq(reduction_numel % split, 0))

    def wrapper_fn(index, reduction_index):
        reduction_index, = reduction_index
        *new_index, reduction_block = index
        indices = block_size * reduction_block + reduction_index

        def body():
            return loader(new_index, reindex([indices]))
        if need_mask:
            mask = ops.lt(ops.index_expr(indices, torch.int32), ops.index_expr(reduction_numel, torch.int32))
            return ops.masked(mask, body, default)
        else:
            return body()
    return wrapper_fn