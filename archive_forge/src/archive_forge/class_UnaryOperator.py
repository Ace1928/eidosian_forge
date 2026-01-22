import operator as op
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class UnaryOperator(AffAtom):
    """
    Base class for expressions involving unary operators.
    """

    def __init__(self, expr) -> None:
        super(UnaryOperator, self).__init__(expr)

    def name(self):
        return self.OP_NAME + self.args[0].name()

    def numeric(self, values):
        return self.OP_FUNC(values[0])