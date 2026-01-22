from pyomo.contrib.pynumero.sparse.block_vector import BlockVector
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
from scipy.sparse import isspmatrix
from .base_block import BaseBlockMatrix
import operator
import numpy as np
import logging
import warnings
def _comparison_helper(self, operation, other):
    result = BlockMatrix(self.bshape[0], self.bshape[1])
    if isinstance(other, BlockMatrix) and other.bshape == self.bshape:
        m, n = self.bshape
        for i in range(m):
            for j in range(n):
                if not self.is_empty_block(i, j) and (not other.is_empty_block(i, j)):
                    result.set_block(i, j, operation(self._blocks[i, j], other.get_block(i, j)))
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = coo_matrix((nrows, ncols))
                    if not self.is_empty_block(i, j):
                        result.set_block(i, j, operation(self._blocks[i, j], mat))
                    elif not other.is_empty_block(i, j):
                        result.set_block(i, j, operation(mat, other.get_block(i, j)))
                    else:
                        result.set_block(i, j, operation(mat, mat))
        return result
    elif isinstance(other, BlockMatrix) or isspmatrix(other):
        if isinstance(other, BlockMatrix):
            raise NotImplementedError('Operation supported with same block structure only')
        else:
            raise NotImplementedError('Operation not supported by BlockMatrix')
    elif np.isscalar(other):
        m, n = self.bshape
        for i in range(m):
            for j in range(n):
                if not self.is_empty_block(i, j):
                    result.set_block(i, j, operation(self._blocks[i, j], other))
                else:
                    nrows = self._brow_lengths[i]
                    ncols = self._bcol_lengths[j]
                    mat = coo_matrix((nrows, ncols))
                    result.set_block(i, j, operation(mat, other))
        return result
    else:
        if other.__class__.__name__ == 'MPIBlockMatrix':
            raise RuntimeError('Operation not supported by BlockMatrix')
        raise NotImplementedError('Operation not supported by BlockMatrix')