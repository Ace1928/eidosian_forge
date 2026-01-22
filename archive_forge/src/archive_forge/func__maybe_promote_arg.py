from __future__ import annotations
import abc
import dataclasses
import inspect
import logging
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Set
import torch
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch._subclasses import fake_tensor
from torch.fx.experimental import proxy_tensor
from torch.fx.node import Node  # noqa: F401
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import _pass, diagnostics, type_utils as fx_type_utils
from torch.utils import _python_dispatch, _pytree
@_beartype.beartype
def _maybe_promote_arg(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, fx_arg: torch.fx.node.Argument, dtype: Optional[torch.dtype]) -> torch.fx.node.Argument:
    """Promote fx_arg to dtype if necessary."""
    if dtype is None:
        diagnostic.info('Argument %s is not promoted. Not mentioned by type promotion rule.', fx_arg)
        return fx_arg
    if isinstance(fx_arg, torch.fx.Node):
        arg_val = self.env[fx_arg]
        if isinstance(arg_val, torch.Tensor):
            if (old_dtype := arg_val.dtype) != dtype:
                graph = node.graph
                with graph.inserting_before(node):
                    diagnostic.info('Argument %s(%s) is promoted to %s.', fx_arg, old_dtype, dtype)
                    return self._create_node(graph, 'call_function', torch.ops.prims.convert_element_type.default, (fx_arg,), {'dtype': dtype})
            diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
            return fx_arg
        elif fx_type_utils.is_torch_symbolic_type(arg_val):
            arg_type = type(arg_val)
            equivalent_dtype = fx_type_utils.from_scalar_type_to_torch_dtype(arg_type)
            assert equivalent_dtype is not None, f'Unexpected arg_type: {arg_type}'
            if equivalent_dtype != dtype:
                graph = node.graph
                with graph.inserting_before(node):
                    diagnostic.info('Argument %s(Scalar of equivalent dtype: %s) is promoted to %s.', fx_arg, equivalent_dtype, dtype)
                    return self._create_node(graph, 'call_function', torch.ops.aten.scalar_tensor.default, (fx_arg,), {'dtype': dtype})
            diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
            return fx_arg
    elif (equivalent_dtype := fx_type_utils.from_scalar_type_to_torch_dtype(type(fx_arg))) is not None:
        if equivalent_dtype != dtype:
            graph = node.graph
            with graph.inserting_before(node):
                diagnostic.info('Argument %s(Scalar of equivalent dtype: %s) is promoted to %s.', fx_arg, equivalent_dtype, dtype)
                return self._create_node(graph, 'call_function', torch.ops.aten.scalar_tensor.default, (fx_arg,), {'dtype': dtype})
        diagnostic.info('Argument %s is not promoted. Already %s.', fx_arg, dtype)
        return fx_arg
    elif isinstance(fx_arg, (tuple, list)):
        diagnostic.info('Argument %s is a tuple/list. Promoting each element.', fx_arg)
        return type(fx_arg)((self._maybe_promote_arg(diagnostic, node, fx_arg_elem, dtype) for fx_arg_elem in fx_arg))
    raise NotImplementedError(f'Unknown fx arg type: {type(fx_arg)}')