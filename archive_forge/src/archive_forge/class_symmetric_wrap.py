from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class symmetric_wrap(Wrap):
    """Asserts that a real square matrix is symmetric
    """

    def validate_arguments(self) -> None:
        validate_real_square(self.args[0])

    def is_symmetric(self) -> bool:
        return True

    def is_hermitian(self) -> bool:
        return True