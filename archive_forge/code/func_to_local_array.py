from __future__ import annotations
from pyomo.common.dependencies import mpi4py
from .mpi_block_vector import MPIBlockVector
from .block_vector import BlockVector
from .block_matrix import BlockMatrix, NotFullyDefinedBlockMatrixError
from .block_matrix import assert_block_structure as block_matrix_assert_block_structure
from .base_block import BaseBlockMatrix
import numpy as np
from scipy.sparse import coo_matrix
import operator
def to_local_array(self):
    """
        This method is only for testing/debugging

        Returns
        -------
        result: np.ndarray
        """
    assert_block_structure(self)
    local_result = self._block_matrix.copy_structure()
    rank = self._mpiw.Get_rank()
    block_indices = self._unique_owned_mask if rank != 0 else self._owned_mask
    ii, jj = np.nonzero(block_indices)
    for i, j in zip(ii, jj):
        if not self._block_matrix.is_empty_block(i, j):
            local_result.set_block(i, j, self.get_block(i, j))
    local_result = local_result.toarray()
    global_result = np.zeros(shape=self.shape, dtype=local_result.dtype)
    self._mpiw.Allreduce(local_result, global_result)
    return global_result