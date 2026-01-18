import copy
import dataclasses
import functools
import io
import json
import pathlib
import re
import sys
import types
import warnings
import weakref
import zipfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import patch
import sympy
import torch
import torch._dynamo
import torch.fx
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._decomp import core_aten_decompositions, get_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.source import ConstantSource
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._functorch.aot_autograd import aot_export_module, GraphSignature
from torch._functorch.eager_transforms import functionalize
from torch._guards import detect_fake_mode
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.export import _create_constraint, _Dim, Constraint
from torch.export.exported_program import (
from torch.export.graph_signature import (
from torch.fx import traceback as fx_traceback
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode
from torch.fx.experimental.symbolic_shapes import (
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges
from .exported_program import (
from .passes.add_runtime_assertions_for_constraints_pass import (
from .passes.lift_constant_tensor_pass import lift_constant_tensor_pass
from .passes.remove_runtime_assertions import _RemoveRuntimeAssertionsPass
from .passes.replace_sym_size_ops_pass import _replace_sym_size_ops_pass
from .passes.replace_view_ops_with_view_copy_ops_pass import (
from .wrappers import _wrap_submodules
def tree_zip(combined_args, dynamic_shapes):
    if isinstance(combined_args, (tuple, list)):
        if not isinstance(dynamic_shapes, Sequence):
            raise UserError(UserErrorType.INVALID_INPUT, f'Expected dynamic_shapes of a {type(combined_args)} to be a Sequence, got {dynamic_shapes} instead')
        if len(combined_args) != len(dynamic_shapes):
            raise UserError(UserErrorType.INVALID_INPUT, f'Expected {dynamic_shapes} to have {len(combined_args)} items')
        for i, shape in enumerate(dynamic_shapes):
            yield from tree_zip(combined_args[i], shape)
    elif isinstance(combined_args, dict):
        if not isinstance(dynamic_shapes, Mapping):
            raise UserError(UserErrorType.INVALID_INPUT, f'Expected dynamic_shapes of a {type(combined_args)} to be a Mapping, got {dynamic_shapes} instead')
        if len(combined_args) != len(dynamic_shapes):
            raise UserError(UserErrorType.INVALID_INPUT, f'Expected {dynamic_shapes} to have {len(combined_args)} items')
        for k, shape in dynamic_shapes.items():
            yield from tree_zip(combined_args[k], shape)
    elif dataclasses.is_dataclass(combined_args):
        if not type(dynamic_shapes) == type(combined_args):
            raise UserError(UserErrorType.INVALID_INPUT, f'Expected dynamic_shapes of a {type(combined_args)} to be a {type(combined_args)}, got {dynamic_shapes} instead')
        for f in dataclasses.fields(combined_args):
            yield from tree_zip(getattr(combined_args, f.name), getattr(dynamic_shapes, f.name))
    elif isinstance(combined_args, torch.Tensor):
        yield (combined_args, dynamic_shapes)
    elif dynamic_shapes is not None:
        raise UserError(UserErrorType.INVALID_INPUT, f'Expected dynamic_shapes of a {type(combined_args)} to be None, got {dynamic_shapes} instead')