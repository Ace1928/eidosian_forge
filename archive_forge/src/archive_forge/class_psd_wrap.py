from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class psd_wrap(Wrap):
    """Asserts that a square matrix is PSD.
    """

    def validate_arguments(self) -> None:
        arg = self.args[0]
        ndim_test = len(arg.shape) == 2
        if not ndim_test:
            raise ValueError('The input must be a square matrix.')
        elif arg.shape[0] != arg.shape[1]:
            raise ValueError('The input must be a square matrix.')

    def is_psd(self) -> bool:
        return True

    def is_nsd(self) -> bool:
        return False

    def is_symmetric(self) -> bool:
        return not self.args[0].is_complex()

    def is_hermitian(self) -> bool:
        return True