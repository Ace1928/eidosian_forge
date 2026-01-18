import warnings
from typing import List, Tuple
import numpy as np
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def validate_arguments(self) -> None:
    """Checks that both arguments are vectors, and the first is constant.
        """
    if not self.args[0].ndim <= 1 or not self.args[1].ndim <= 1:
        raise ValueError('The arguments to conv must be scalar or 1D.')
    if not self.args[0].is_constant():
        raise ValueError('The first argument to conv must be constant.')