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
def _disallow_eval_train(model: GraphModule):
    """
    Disallow calling `model.train()` or `model.eval()` on the given GraphModule.
    This is useful for exported models, where these methods don't actually behave as expected.
    """

    def _train(self, mode: bool=True):
        raise NotImplementedError('Calling train() is not supported yet.')

    def _eval(self, mode: bool=True):
        raise NotImplementedError('Calling eval() is not supported yet.')
    model.train = types.MethodType(_train, model)
    model.eval = types.MethodType(_eval, model)
    return model