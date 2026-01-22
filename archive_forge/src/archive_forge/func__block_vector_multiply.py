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
def _block_vector_multiply(self, x):
    """
        In this method, we assume that we can access the correct blocks from x. This means that
        _get_block_vector_for_dot_product should be called first.

        For a given block row, if there are multiple non-empty blocks with different rank owners,
        then the result for that row is owned by all, and we need to do an Allreduce. Otherwise the
        rank owner of the resulting block is the rank owner of the non-empty blocks in the block row.
        """
    n_block_rows, n_block_cols = self.bshape
    comm = self._mpiw
    rank = comm.Get_rank()
    blocks_that_need_reduced = np.zeros(n_block_rows, dtype=np.int64)
    res_rank_owner = np.zeros(n_block_rows, dtype=np.int64)
    for i, j in zip(*np.nonzero(self._block_matrix._block_mask)):
        blocks_that_need_reduced[i] = 1
        res_rank_owner[i] = self._rank_owner[i, j]
    local_empty_rows = self._block_matrix._block_mask.any(axis=1)
    local_empty_rows = np.array(local_empty_rows, dtype=np.int64)
    global_empty_rows = np.empty(local_empty_rows.size, dtype=np.int64)
    comm.Allreduce(local_empty_rows, global_empty_rows)
    empty_rows = np.nonzero(global_empty_rows == 0)[0]
    global_blocks_that_need_reduced = np.zeros(n_block_rows, dtype=np.int64)
    comm.Allreduce(blocks_that_need_reduced, global_blocks_that_need_reduced)
    block_indices_that_need_reduced = np.nonzero(global_blocks_that_need_reduced > 1)[0]
    global_res_rank_owner = np.zeros(n_block_rows, dtype=np.int64)
    comm.Allreduce(res_rank_owner, global_res_rank_owner)
    global_res_rank_owner[block_indices_that_need_reduced] = -1
    for ndx in empty_rows:
        row_owners = set(self._rank_owner[ndx, :])
        if len(row_owners) == 1:
            global_res_rank_owner[ndx] = row_owners.pop()
        elif len(row_owners) == 2 and -1 in row_owners:
            tmp = row_owners.pop()
            if tmp == -1:
                global_res_rank_owner[ndx] = row_owners.pop()
            else:
                global_res_rank_owner[ndx] = tmp
        else:
            global_res_rank_owner[ndx] = -1
    res = MPIBlockVector(nblocks=n_block_rows, rank_owner=global_res_rank_owner, mpi_comm=comm, assert_correct_owners=False)
    for ndx in np.nonzero(res.ownership_mask)[0]:
        res.set_block(ndx, np.zeros(self.get_row_size(ndx)))
    if rank == 0:
        block_indices = self._owned_mask
    else:
        block_indices = self._unique_owned_mask
    block_indices = np.bitwise_and(block_indices, self._block_matrix._block_mask)
    for row_ndx, col_ndx in zip(*np.nonzero(block_indices)):
        res_blk = res.get_block(row_ndx)
        tmp = self.get_block(row_ndx, col_ndx) * x.get_block(col_ndx)
        tmp += res_blk
        res.set_block(row_ndx, tmp)
    for ndx in block_indices_that_need_reduced:
        local = res.get_block(ndx)
        flat_local = local.flatten()
        flat_global = np.zeros(flat_local.size)
        comm.Allreduce(flat_local, flat_global)
        if isinstance(local, BlockVector):
            local.copyfrom(flat_global)
        else:
            res.set_block(ndx, flat_global)
    return res