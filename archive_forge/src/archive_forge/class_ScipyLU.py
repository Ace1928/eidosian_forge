from .base import (
from scipy.sparse.linalg import splu, LinearOperator
from scipy.linalg import eigvals
from scipy.sparse import isspmatrix_csc, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
import numpy as np
from typing import Union, Tuple, Optional, Callable
class ScipyLU(DirectLinearSolverInterface):

    def __init__(self):
        self._lu = None

    def do_symbolic_factorization(self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool=True) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res

    def do_numeric_factorization(self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool=True) -> LinearSolverResults:
        if not isspmatrix_csc(matrix):
            matrix = matrix.tocsc()
        res = LinearSolverResults()
        try:
            self._lu = splu(matrix)
            res.status = LinearSolverStatus.successful
        except RuntimeError as err:
            if raise_on_error:
                raise err
            if 'Factor is exactly singular' in str(err):
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res

    def do_back_solve(self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
        else:
            _rhs = rhs
        result = self._lu.solve(_rhs)
        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result
        return (result, LinearSolverResults(LinearSolverStatus.successful))