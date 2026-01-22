import operator
import traceback
import typing
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from functorch.experimental.control_flow import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree
def _fx(self, kind: str, target: torch.fx.node.Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument], meta: NodeMetadata) -> ProxyValue:
    args_data, kwargs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, (args, kwargs))
    res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
    args_proxy, kwargs_proxy = pytree.tree_map_only(ProxyValue, lambda x: x.proxy, (args, kwargs))
    name = None
    if isinstance(target, torch._ops.OpOverload):
        name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)
    res_proxy = self.tracer.create_proxy(kind, target, args_proxy, kwargs_proxy, name=name)
    res_proxy.node.meta.update(meta.data)
    self.tracer.set_metadata(res_proxy.node, res_data)
    return ProxyValue(res_data, res_proxy)