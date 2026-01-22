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
def _inplace_binary_operation_helper(self, other, operation):
    if isinstance(other, (MPIBlockMatrix, BlockMatrix)):
        assert operation in {operator.iadd, operator.isub}
        assert other.bshape == self.bshape, 'dimensions mismatch {} != {}'.format(self.bshape, other.bshape)
        if isinstance(other, MPIBlockMatrix):
            assert np.array_equal(self._rank_owner, other._rank_owner), 'MPIBlockMatrices must be distributed in same processors'
        block_indices = other.get_block_mask(copy=False)
        block_indices = np.bitwise_and(block_indices, self._owned_mask)
        ii, jj = np.nonzero(block_indices)
        for i, j in zip(ii, jj):
            mat1 = self.get_block(i, j)
            mat2 = other.get_block(i, j)
            if mat1 is not None and mat2 is not None:
                mat1 = operation(mat1, mat2)
                self.set_block(i, j, mat1)
            elif mat1 is None and mat2 is not None:
                if operation is operator.iadd:
                    sub_res = mat2.copy()
                else:
                    sub_res = -mat2
                self.set_block(i, j, sub_res)
            else:
                raise RuntimeError('Please report this to the developers.')
    elif np.isscalar(other):
        block_indices = np.bitwise_and(self.get_block_mask(copy=False), self._owned_mask)
        for i, j in zip(*np.nonzero(block_indices)):
            blk = self.get_block(i, j)
            blk = operation(blk, other)
            self.set_block(i, j, blk)
    else:
        raise NotImplementedError('Operation not supported by MPIBlockMatrix')
    return self