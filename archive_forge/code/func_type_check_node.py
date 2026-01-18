from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def type_check_node(self, n: Node):
    """
        Type check a given fx node.
        Current operations:
        - Reshape
        - Transpose
        - Add
        - Relu
        - conv2d
        - batchnorm2d
        - flatten
        - maxpool2d
        - adaptiveavgpool2d
        - linear
        """
    if n.type is None:
        n.type = Dyn
    if n.op == 'placeholder':
        return n.type
    elif n.op == 'get_attr':
        t = get_parameter(self.traced, n.target)
        if isinstance(t.data, torch.Tensor):
            n.type = TensorType(t.data.shape)
        return n.type
    elif n.op == 'call_function':
        if n.target == getattr:
            assert getattr in _INFERENCE_RULES
            return _INFERENCE_RULES[n.target](n, self.traced)
        elif n.target in _INFERENCE_RULES:
            return _INFERENCE_RULES[n.target](n)
        else:
            raise RuntimeError(f'No inference rule registered for target {n.target}!')
    elif n.op == 'call_module':
        module_instance = self.traced.get_submodule(n.target)
        if type(module_instance) in _INFERENCE_RULES:
            return _INFERENCE_RULES[type(module_instance)](n, module_instance)
        else:
            raise RuntimeError(f'No inference rule registered for class {type(module_instance)}!')
    elif n.op == 'output':

        def get_node_type(a):
            return a.type
        n.type = torch.fx.node.map_arg(n.args[0], get_node_type)
        return n.type
    else:
        raise NotImplementedError(f'Method {n.op} not yet implemented')