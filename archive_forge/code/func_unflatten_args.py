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
def unflatten_args(new_tensor_args, new_non_tensor_args):
    result = []
    it_tensors = iter(new_tensor_args)
    it_non_tensors = iter(new_non_tensor_args)
    for is_tensor in is_arg_tensor:
        if is_tensor:
            result.append(next(it_tensors))
        else:
            result.append(next(it_non_tensors))
    r = pytree.tree_unflatten(result, args_spec)
    return (r.get('args', []), r.get('kwargs', {}))