from torch.fx.experimental.migrate_gradual_types.constraint import Conj, Disj, T, F, BinConstraintT, BVar, is_bool_expr
from torch.fx.experimental.migrate_gradual_types.constraint import BinConstraintD, TVar, DVar
from torch.fx.experimental.migrate_gradual_types.constraint import Prod, is_algebraic_expression, is_dim
from torch.fx.experimental.migrate_gradual_types.constraint_generator import ConstraintGenerator
from torch.fx.experimental.migrate_gradual_types.constraint_transformation import transform_constraint
from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_eq, op_neq, op_gt, op_lt
from torch.fx.experimental.migrate_gradual_types.operation import op_leq, op_sub, op_div, op_mul, op_mod
from torch.fx.tensor_type import TensorType, Dyn
def transform_var(tensor, counter, dimension_dict):
    """
        Transforms tensor variables to a format understood by z3
        Args:
            tensor: Tensor variable or a tensor type potentially with variable dimensions
        Returns: Transformed variable to a z3 format

        """
    if isinstance(tensor, TensorType):
        res = []
        for t in tensor.__args__:
            transformed, counter = transform_dimension(t, counter, dimension_dict)
            res.append(transformed)
        assert len(res) <= 4
        if len(tensor.__args__) == 1:
            return (tensor_type.tensor1(res[0]), counter)
        elif len(tensor.__args__) == 2:
            return (tensor_type.tensor2(res[0], res[1]), counter)
        elif len(tensor.__args__) == 3:
            return (tensor_type.tensor3(res[0], res[1], res[2]), counter)
        elif len(tensor.__args__) == 4:
            return (tensor_type.tensor4(res[0], res[1], res[2], res[3]), counter)
    elif tensor == Dyn:
        return (z3_dyn, counter)
    elif isinstance(tensor, TVar):
        return (z3.Const(tensor.tvar, tensor_type), counter)