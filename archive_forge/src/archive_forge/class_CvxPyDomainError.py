from __future__ import division
import warnings
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import is_sparse
from cvxpy.utilities.linalg import sparse_cholesky
class CvxPyDomainError(Exception):
    pass