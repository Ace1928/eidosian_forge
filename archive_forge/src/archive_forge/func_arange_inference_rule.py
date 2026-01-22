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
@register_inference_rule(torch.arange)
def arange_inference_rule(n: Node, symbols, constraints, counter):
    start = 0
    step = 1
    if len(n.args) == 1:
        end = symbols[n.args[0]]
    else:
        raise NotImplementedError('Not yet implemented')
    d1, counter = gen_dvar(counter)
    size_constraint = BinConstraintD(d1, BinConstraintD(BinConstraintD(end, start, op_sub), step, op_div), op_eq)
    arange, counter = gen_tvar(counter)
    symbols[n] = arange
    c1 = Disj([BinConstraintD(end, Dyn, op_eq), BinConstraintD(start, Dyn, op_eq), BinConstraintD(step, Dyn, op_eq)])
    c2 = BinConstraintD(d1, Dyn, op_eq)
    both_dyn = Conj([c1, c2])
    c11 = Conj([BinConstraintD(end, Dyn, op_neq), BinConstraintD(start, Dyn, op_neq), BinConstraintD(step, Dyn, op_neq)])
    c22 = BinConstraintD(d1, Dyn, op_neq)
    both_numbers = Conj([c11, c22, size_constraint])
    return ([BinConstraintT(arange, TensorType([d1]), op_eq), Disj([both_dyn, both_numbers])], counter)