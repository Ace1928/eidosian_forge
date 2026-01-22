from collections import defaultdict
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.cvxcore.python import canonInterface
class ReducedMat:
    """Utility class for condensing the mapping from parameters to problem data.

    For maximum efficiency of representation and application, the mapping from
    parameters to problem data must be condensed. It begins as a CSC sparse matrix
    matrix_data, such that multiplying by a parameter vector gives the problem data.
    The row index array and column pointer array are saved as problem_data_index,
    and a CSR matrix reduced_mat that when multiplied by a parameter vector gives
    the values array. The ReducedMat class caches the condensed representation
    and provides a method for multiplying by a parameter vector.

    This class consolidates code from ParamConeProg and ParamQuadProg.

    Attributes
    ----------
    matrix_data : SciPy CSC sparse matrix
       A matrix representing the mapping from parameter to problem data.
    var_len : int
       The length of the problem variable.
    quad_form: (optional) if True, consider quadratic form matrix P
    """

    def __init__(self, matrix_data, var_len: int, quad_form: bool=False) -> None:
        self.matrix_data = matrix_data
        self.var_len = var_len
        self.quad_form = quad_form
        self.reduced_mat = None
        self.problem_data_index = None
        self.mapping_nonzero = None

    def cache(self, keep_zeros: bool=False) -> None:
        """Cache computed attributes if not present.

        Parameters
        ----------
            keep_zeros: (optional) if True, store explicit zeros in A where
                        parameters are affected.
        """
        if self.matrix_data is None:
            return
        if self.reduced_mat is None:
            if np.prod(self.matrix_data.shape) != 0:
                reduced_mat, indices, indptr, shape = canonInterface.reduce_problem_data_tensor(self.matrix_data, self.var_len, self.quad_form)
                self.reduced_mat = reduced_mat
                self.problem_data_index = (indices, indptr, shape)
            else:
                self.reduced_mat = self.matrix_data
                self.problem_data_index = None
        if keep_zeros and self.mapping_nonzero is None:
            self.mapping_nonzero = canonInterface.A_mapping_nonzero_rows(self.matrix_data, self.var_len)

    def get_matrix_from_tensor(self, param_vec: np.ndarray, with_offset: bool=True) -> Tuple:
        """Wraps get_matrix_from_tensor in canonInterface.

        Parameters
        ----------
            param_vec: flattened parameter vector
            with_offset: (optional) return offset. Defaults to True.

        Returns
        -------
            A tuple (A, b), where A is a matrix with `var_length` columns
            and b is a flattened NumPy array representing the constant offset.
            If with_offset=False, returned b is None.
        """
        return canonInterface.get_matrix_from_tensor(self.reduced_mat, param_vec, self.var_len, nonzero_rows=self.mapping_nonzero, with_offset=with_offset, problem_data_index=self.problem_data_index)