import torch
import operator
import warnings
from typing import Callable, Dict, Iterable
from torch.fx._symbolic_trace import _assert_is_none
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, CalcProduct, \
from torch.fx.experimental.migrate_gradual_types.operation import \
from torch.fx.node import Target, Node
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar, gen_tvar, \
from torch.fx.tensor_type import Dyn, TensorType
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
@register_inference_rule('type_as')
def type_inference_rule(n: Node, symbols, constraints, counter):
    """
    We generate the constraint: input = output
    """
    assert isinstance(n.args[0], Node)
    assert isinstance(n.args[1], Node)
    output, counter = gen_tvar(counter)
    symbols[n] = output
    from_arg = symbols[n.args[0]]
    to_arg = symbols[n.args[1]]
    assert isinstance(from_arg, TVar)
    assert isinstance(to_arg, TVar)
    return ([BinConstraintT(from_arg, to_arg, op_consistency), BinConstraintT(output, to_arg, op_eq)], counter)