from __future__ import annotations, division
import operator
from typing import List
import numpy as np
import scipy.sparse as sp
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.lin_ops.lin_op import NO_OP, LinOp
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.utilities.replace_quad_forms import (
Stack q with constant offset as last row.

        Args:
            q_list: list of q submatrices as SciPy sparse matrices or NumPy arrays.
            constant: The constant offset as a CSC sparse matrix.
            num_params: number of parameters in the problem.

        Returns:
            A CSR sparse representation of the merged q matrix.
        