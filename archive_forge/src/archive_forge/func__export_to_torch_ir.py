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
def _export_to_torch_ir(f: Callable, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]]=None, constraints: Optional[List[Constraint]]=None, *, preserve_module_call_signature: Tuple[str, ...]=(), disable_constraint_solver: bool=False) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """
    constraints = constraints or []
    kwargs = kwargs or {}
    if not isinstance(args, tuple):
        raise UserError(UserErrorType.INVALID_INPUT, f'Expecting `args` to be a tuple of example positional inputs, got {type(args)}')
    if isinstance(f, ExportedProgram):
        f = f.module()
    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
        try:
            module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
            with _wrap_submodules(f, preserve_module_call_signature, module_call_specs):
                gm_torch_level, _ = torch._dynamo.export(f, constraints=constraints, assume_static_by_default=True, tracing_mode='symbolic', disable_constraint_solver=disable_constraint_solver)(*args, **kwargs)
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(UserErrorType.ANTI_PATTERN, f'Consider annotating your code using torch._constrain_as_*(). {str(e)}', case_name='constrain_as_size_example')
    gm_torch_level.meta['module_call_specs'] = module_call_specs
    return gm_torch_level