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
def _assert_correct_owners(self, root=0):
    rank = self._mpiw.Get_rank()
    num_processors = self._mpiw.Get_size()
    if num_processors == 1:
        return True
    local_owners = self._rank_owner.flatten()
    flat_size = self.bshape[0] * self.bshape[1]
    receive_data = None
    if rank == root:
        receive_data = np.empty(flat_size * num_processors, dtype=np.int64)
    self._mpiw.Gather(local_owners, receive_data, root=root)
    if rank == root:
        owners_in_processor = np.split(receive_data, num_processors)
        root_rank_owners = owners_in_processor[root]
        for k in range(num_processors):
            if k != root:
                if not np.array_equal(owners_in_processor[k], root_rank_owners):
                    return False
    return True