import operator
import types
import torch
from torch._export import capture_pre_autograd_graph
from torch.fx import (
from torch.nn.utils.fusion import fuse_conv_bn_weights
from typing import Any, Callable, Dict, Optional, Tuple, List, Union
from torch.utils._pytree import LeafSpec
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.ao.quantization.quantizer import QuantizationAnnotation
def _find_q_dq_node_for_user(produer: torch.fx.Node, user: torch.fx.Node) -> Tuple[Any, Any]:
    """
    Find q, dq pair corresponding to [producer -> q -> dq -> user]
    Utils works by finding dq arg of user and ensuring it is connected to
    producer
    """
    dq_node = None
    for n in user.args:
        if isinstance(n, torch.fx.Node) and n.op == 'call_function' and (n.target in _DEQUANTIZE_OPS):
            if _is_connected(produer, n):
                dq_node = n
                break
    if dq_node is None:
        for n in user.kwargs:
            if isinstance(n, torch.fx.Node) and n.op == 'call_function' and (n.target in _DEQUANTIZE_OPS):
                if _is_connected(produer, n):
                    dq_node = n
                    break
    if dq_node is None:
        return (None, None)
    q_node = None
    if dq_node.args[0].op == 'call_function' and dq_node.args[0].target in _QUANTIZE_OPS:
        q_node = dq_node.args[0]
    return (q_node, dq_node)