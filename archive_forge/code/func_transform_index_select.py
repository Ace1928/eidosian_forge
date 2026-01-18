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
@register_transformation_rule(IndexSelect)
def transform_index_select(constraint, counter):
    """
    The constraints consider the given tensor size, checks if the index is valid
    and if so, generates a constraint for replacing the input dimension
    with the required dimension
    """
    dims, counter = gen_tensor_dims(constraint.tensor_size, counter)
    is_valid_index = valid_index(constraint.index, dims)
    nat_constraints = gen_nat_constraints(dims)
    if is_valid_index == T():
        new_dims = copy.deepcopy(dims)
        new_dims[constraint.index] = constraint.dim_replace
    transformed_constraint = Conj([BinConstraintT(constraint.input_var, TensorType(dims), op_eq), *nat_constraints, is_valid_index, BinConstraintT(constraint.output, TensorType(new_dims), op_eq)])
    return (transformed_constraint, counter)