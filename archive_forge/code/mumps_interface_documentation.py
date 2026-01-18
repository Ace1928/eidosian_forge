from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix

        Perform back solve with Mumps. Note that both do_symbolic_factorization and
        do_numeric_factorization should be called before do_back_solve.

        Parameters
        ----------
        rhs: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The right hand side in matrix * x = rhs.

        Returns
        -------
        result: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The x in matrix * x = rhs. If rhs is a BlockVector, then, result
            will be a BlockVector with the same block structure as rhs.
        