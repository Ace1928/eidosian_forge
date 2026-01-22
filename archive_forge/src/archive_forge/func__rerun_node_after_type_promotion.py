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
def _rerun_node_after_type_promotion(self, diagnostic: diagnostics.Diagnostic, node: torch.fx.Node, expected_out_dtype: torch.dtype) -> None:
    """Rerun a node after type promotion and update node.meta["val"] with the output value."""
    node_val = node.meta.get('val', None)
    assert node_val is not None, f"Node {node} node.meta['val'] is not set."
    args, kwargs = self.fetch_args_kwargs_from_env(node)
    target = node.target
    assert isinstance(target, torch._ops.OpOverload), f'Expected OpOverload, got {type(target)}'
    node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)
    new_node_val = self._run_node_and_set_meta(node)
    assert isinstance(new_node_val, type(node_val)), f'run_node output type should not change between runs. Got {type(new_node_val)}, expect {type(node_val)}.'
    if isinstance(node_val, torch.Tensor):
        prev_node_dtype = node_val.dtype
        assert prev_node_dtype == expected_out_dtype, f"node.meta['val'].dtype({prev_node_dtype}) does not agree with type promotion rule({expected_out_dtype})."
        if new_node_val.dtype != expected_out_dtype:
            graph = node.graph
            with graph.inserting_after(node):
                output_cast_node = self._create_node(graph, 'call_function', torch.ops.prims.convert_element_type.default, (node,), {'dtype': expected_out_dtype})
                node.replace_all_uses_with(output_cast_node)
                output_cast_node.args = (node,)
                diagnostic.info("Node '%s' output dtype becomes %s due to op math. Cast back to %s.", node, new_node_val.dtype, expected_out_dtype)
    elif fx_type_utils.is_torch_symbolic_type(node_val):
        raise NotImplementedError('Type promotion does not support node output of sym types.')
    elif isinstance(node_val, (list, tuple)):
        raise NotImplementedError('Type promotion does not support node output of list or tuple.')
    else:
        raise RuntimeError(f'Unexpected node output type: {type(node_val)}.')