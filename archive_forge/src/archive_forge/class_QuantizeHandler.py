from abc import ABC
from typing import Callable, Dict, List, Optional, Type
import torch
from torch.ao.quantization.backend_config import (
from torch.ao.quantization.utils import NodePattern, Pattern, QuantizerCls
from torch.fx.graph import Node
from .utils import all_node_args_have_no_tensors
class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """

    def __init__(self, node_pattern: NodePattern, modules: Dict[str, torch.nn.Module], root_node_getter: Optional[Callable]=None, is_custom_module=False, is_standalone_module=False):
        """ Records pattern information in __init__, which will be used
        in convert
        """
        self.node_pattern = node_pattern
        self.modules = modules
        if root_node_getter is None:
            root_node_getter = _default_root_node_getter
        self.root_node = root_node_getter(node_pattern)
        self.is_custom_module_ = is_custom_module
        self.is_standalone_module_ = is_standalone_module
        self.num_tensor_args = 0
        if isinstance(self.root_node, Node):
            cache_for_no_tensor_check: Dict[Node, bool] = {}
            for arg_idx in range(len(self.root_node.args)):
                arg = self.root_node.args[arg_idx]
                if isinstance(arg, Node) and (not all_node_args_have_no_tensors(arg, self.modules, cache_for_no_tensor_check)):
                    self.num_tensor_args += 1

    def is_general_tensor_value_op(self) -> bool:
        """
        Returns True if the operator works for both floating point and
        quantized input, and does some computation based on the input Tensor,
        or the ops that only re-arranges the Tensor values or query some metadata
        about the Tensor
        so we need to insert observer/fake_quant for the output of the
        operator (same observer instance as input)
        since the distribution of values is different for input and output
        Tensors (for HistogramObserver) while they share the same quantization
        parameters
        Example operator: avgpool2d, reshape, transpose, maxpool2d
        Example observed operator:
        observer_0 - avgpool2d - observer_0 (same observer instance as input)
        """
        return False

    def is_custom_module(self):
        return self.is_custom_module_

    def is_standalone_module(self):
        return self.is_standalone_module_