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
def _is_conv(n: Node):
    """
    Return whether the node refers to an aten conv op.
    """
    return n.op == 'call_function' and n.target in [torch.ops.aten.conv1d.default, torch.ops.aten.conv2d.default]