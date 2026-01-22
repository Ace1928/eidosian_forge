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
class ExportInterpreter(fx.Interpreter):
    """
        Interpreter to callback on any _ExportPassBase functions
        """

    def __init__(self, callback: '_ExportPassBase', gm: fx.GraphModule) -> None:
        super().__init__(gm)
        self.callback = callback
        self.node: torch.fx.Node = next(iter(gm.graph.nodes))

    def placeholder(self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> ProxyValue:
        arg = super().placeholder(target, args, kwargs)
        return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

    def output(self, target: torch.fx.node.Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> ProxyValue:
        return self.callback.output(args[0], NodeMetadata(self.node.meta)).data

    def call_function(self, target: torch.fx.node.Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> ProxyValue:
        meta = NodeMetadata(self.node.meta)
        if target == operator.getitem:
            value, key = args
            return self.callback.call_getitem(value, key, meta)
        elif getattr(target, '__module__', None) == '_operator':
            assert callable(target)
            return self.callback.call_sym(target, args, meta)
        elif isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            return self.callback.call_operator(target, args, kwargs, meta)
        elif target == torch.ops.higher_order.cond:
            pred, true_fn, false_fn, inputs = args
            return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
        elif target == torch.ops.higher_order.map_impl:
            f, num_args, *rest = args
            return self.callback.call_map(f, num_args, list(rest), meta)
        elif isinstance(target, torch._ops.HigherOrderOperator):
            return self.callback._fx('call_function', target, args, kwargs, meta)
        else:
            raise ExportPassBaseError(f'Unsupported target type: {target}')

    def get_attr(self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> Argument:
        return super().get_attr(target, args, kwargs)

    def call_module(self, target: torch.fx.node.Target, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> None:
        raise ExportPassBaseError('call_module is not supported.')

    def call_method(self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]) -> None:
        raise ExportPassBaseError('call_method is not supported.')

    def run_node(self, n: torch.fx.Node) -> Argument:
        self.node = n
        self.callback.node_debug_str = n.format_node()
        return super().run_node(n)