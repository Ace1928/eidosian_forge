import copy
import itertools
from torch.fx.experimental.migrate_gradual_types.constraint_generator import BinConstraintT, MAX_TENSOR_RANK
from torch.fx.experimental.migrate_gradual_types.constraint import T, BinConstraintD, Conj, Constraint, DVar, TVar, \
from torch.fx.experimental.migrate_gradual_types.constraint import Disj, TGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import DGreatestUpperBound
from torch.fx.experimental.migrate_gradual_types.constraint import CalcConv, CalcMaxPool
from torch.fx.experimental.migrate_gradual_types.constraint import CalcProduct, CanReshape
from torch.fx.experimental.migrate_gradual_types.constraint import ApplyBroadcasting, Prod, F, GetItem, GetItemTensor, IndexSelect
from torch.fx.experimental.migrate_gradual_types.operation import op_eq, op_precision, op_leq, op_matching
from torch.fx.experimental.migrate_gradual_types.operation import op_consistency, op_neq
from torch.fx.experimental.migrate_gradual_types.operation import op_mul, op_add, op_sub, op_div, op_mod
from torch.fx.experimental.migrate_gradual_types.util import gen_tensor_dims, gen_nat_constraints, gen_dvar
from torch.fx.tensor_type import TensorType, Dyn
from typing import Callable, Dict, List
@register_transformation_rule(GetItem)
def transform_get_item(constraint, counter):
    """
    generate an equality of the form:
    t = [a1, ..., an]
    then generate constraints that check if the given index is valid
    given this particular tensor size.
    If the index is valid, generate a constraint to get the item
    Note that we already handled the Dyn input case in the previous
    step.
    Args:
        constraint: GetItem which assumes we are getting an item from a tensor (not Dyn)
        counter: variable tracking
    Returns: simplified constraints for GetItem

    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    nat_constraints = gen_nat_constraints(dims)
    is_valid_index = valid_index(constraint.index, dims)
    all_constraints = [BinConstraintT(constraint.input_var, TensorType(dims), op_eq), *nat_constraints, is_valid_index]
    if is_valid_index == T():
        all_constraints.append(BinConstraintD(constraint.res, dims[constraint.index], op_eq))
    return (Conj(all_constraints), counter)