from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
import cvxpy.utilities as u
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import is_power2, pow_high, pow_mid, pow_neg
def _is_const(p) -> bool:
    return isinstance(p, cvxtypes.constant())