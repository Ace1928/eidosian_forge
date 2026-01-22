import dataclasses
import importlib
import logging
from typing import (
from typing_extensions import TypeAlias
import torch
import torch._C
import torch._ops
import torch._prims.executor
import torch.fx
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx._compatibility import compatibility
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree
def _replace_to_copy_with_to(fx_module: torch.fx.GraphModule) -> None:
    for node in fx_module.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload) and node.target.overloadpacket == torch.ops.aten._to_copy:
            is_default_layout = True
            is_on_same_device = True
            is_cast = True
            are_kwargs_supported = True
            if 'layout' in node.kwargs and node.kwargs['layout'] != torch.strided:
                is_default_layout = False
            if 'device' in node.kwargs and node.kwargs['device'] != node.args[0].meta['val'].device:
                is_on_same_device = False
            if 'dtype' not in node.kwargs:
                is_cast = False
            for kwarg in node.kwargs:
                if kwarg not in ['layout', 'device', 'dtype']:
                    are_kwargs_supported = False
            if len(node.args) == 1 and is_default_layout and is_on_same_device and is_cast and are_kwargs_supported:
                node.kwargs = {'dtype': node.kwargs['dtype']}
                node.target = torch.ops.aten.to.dtype
            else:
                raise RuntimeError(f'aten._to_copy must be replaced with other ONNX-supported aten ops.                          args={[arg.meta for arg in node.args]}, kwargs={node.kwargs}')
    fx_module.recompile()