from torch.fx.experimental.migrate_gradual_types.operation import op_add, op_sub, op_mul, op_div, \
from torch.fx.tensor_type import TensorType, Dyn
class BinConstraintD(BinaryConstraint):
    """
    Binary constraints about dimensions
    """

    def __init__(self, lhs, rhs, op):
        assert is_algebraic_expression(lhs) or is_dim(lhs) or is_bool_expr(lhs)
        assert is_algebraic_expression(rhs) or is_dim(rhs) or is_bool_expr(rhs)
        super().__init__(lhs, rhs, op)

    def __eq__(self, other):
        return super().__eq__(other)