from typing import List, Tuple, Union
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities.power_tools import pow_high, pow_mid, pow_neg
Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        