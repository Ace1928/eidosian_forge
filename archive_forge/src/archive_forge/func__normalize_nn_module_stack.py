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
def _normalize_nn_module_stack(gm_torch_level, root_cls):
    root = "L['self']"
    root_key = re.sub('[^a-zA-Z0-9]', '_', root)
    for gm in gm_torch_level.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op in ['placeholder', 'output']:
                continue
            add_root = True
            if (nn_module_stack := node.meta.get('nn_module_stack', {})):
                path, ty = next(iter(nn_module_stack.values()))
                assert issubclass(ty, torch.nn.Module)
                if path == root and ty is root_cls:
                    add_root = False
            if add_root:

                def normalize_path(path):
                    try:
                        parts = []

                        class Path:

                            def __getattr__(self, name):
                                parts.append(name)
                                return self

                            def __getitem__(self, idx):
                                parts.append(str(idx))
                                return self
                        eval(path, {'L': {'self': Path()}})
                        return '.'.join(parts)
                    except Exception:
                        return path
                nn_module_stack = {root_key: (root, root_cls), **nn_module_stack}
                node.meta['nn_module_stack'] = {key: (normalize_path(path), ty) for key, (path, ty) in nn_module_stack.items()}