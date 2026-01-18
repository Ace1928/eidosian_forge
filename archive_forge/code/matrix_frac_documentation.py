from functools import wraps
from typing import List, Tuple
import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint
Quadratic of piecewise affine if x is PWL and P is constant.
        