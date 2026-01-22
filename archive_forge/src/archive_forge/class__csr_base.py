import numpy as np
from ._matrix import spmatrix
from ._base import _spbase, sparray
from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,
from ._sputils import upcast
from ._compressed import _cs_matrix
class _csr_base(_cs_matrix):
    _format = 'csr'

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support an 'axes' parameter because swapping dimensions is the only logical permutation.")
        M, N = self.shape
        return self._csc_container((self.data, self.indices, self.indptr), shape=(N, M), copy=copy)
    transpose.__doc__ = _spbase.transpose.__doc__

    def tolil(self, copy=False):
        lil = self._lil_container(self.shape, dtype=self.dtype)
        self.sum_duplicates()
        ptr, ind, dat = (self.indptr, self.indices, self.data)
        rows, data = (lil.rows, lil.data)
        for n in range(self.shape[0]):
            start = ptr[n]
            end = ptr[n + 1]
            rows[n] = ind[start:end].tolist()
            data[n] = dat[start:end].tolist()
        return lil
    tolil.__doc__ = _spbase.tolil.__doc__

    def tocsr(self, copy=False):
        if copy:
            return self.copy()
        else:
            return self
    tocsr.__doc__ = _spbase.tocsr.__doc__

    def tocsc(self, copy=False):
        idx_dtype = self._get_index_dtype((self.indptr, self.indices), maxval=max(self.nnz, self.shape[0]))
        indptr = np.empty(self.shape[1] + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))
        csr_tocsc(self.shape[0], self.shape[1], self.indptr.astype(idx_dtype), self.indices.astype(idx_dtype), self.data, indptr, indices, data)
        A = self._csc_container((data, indices, indptr), shape=self.shape)
        A.has_sorted_indices = True
        return A
    tocsc.__doc__ = _spbase.tocsc.__doc__

    def tobsr(self, blocksize=None, copy=True):
        if blocksize is None:
            from ._spfuncs import estimate_blocksize
            return self.tobsr(blocksize=estimate_blocksize(self))
        elif blocksize == (1, 1):
            arg1 = (self.data.reshape(-1, 1, 1), self.indices, self.indptr)
            return self._bsr_container(arg1, shape=self.shape, copy=copy)
        else:
            R, C = blocksize
            M, N = self.shape
            if R < 1 or C < 1 or M % R != 0 or (N % C != 0):
                raise ValueError('invalid blocksize %s' % blocksize)
            blks = csr_count_blocks(M, N, R, C, self.indptr, self.indices)
            idx_dtype = self._get_index_dtype((self.indptr, self.indices), maxval=max(N // C, blks))
            indptr = np.empty(M // R + 1, dtype=idx_dtype)
            indices = np.empty(blks, dtype=idx_dtype)
            data = np.zeros((blks, R, C), dtype=self.dtype)
            csr_tobsr(M, N, R, C, self.indptr.astype(idx_dtype), self.indices.astype(idx_dtype), self.data, indptr, indices, data.ravel())
            return self._bsr_container((data, indices, indptr), shape=self.shape)
    tobsr.__doc__ = _spbase.tobsr.__doc__

    def _swap(self, x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x

    def __iter__(self):
        indptr = np.zeros(2, dtype=self.indptr.dtype)
        shape = (1, self.shape[1])
        i0 = 0
        for i1 in self.indptr[1:]:
            indptr[1] = i1 - i0
            indices = self.indices[i0:i1]
            data = self.data[i0:i1]
            yield self.__class__((data, indices, indptr), shape=shape, copy=True)
            i0 = i1

    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(M, N, self.indptr, self.indices, self.data, i, i + 1, 0, N)
        return self.__class__((data, indices, indptr), shape=(1, N), dtype=self.dtype, copy=False)

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSR matrix (column vector).
        """
        M, N = self.shape
        i = int(i)
        if i < 0:
            i += N
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        indptr, indices, data = get_csr_submatrix(M, N, self.indptr, self.indices, self.data, 0, M, i, i + 1)
        return self.__class__((data, indices, indptr), shape=(M, 1), dtype=self.dtype, copy=False)

    def _get_intXarray(self, row, col):
        return self._getrow(row)._minor_index_fancy(col)

    def _get_intXslice(self, row, col):
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        M, N = self.shape
        start, stop, stride = col.indices(N)
        ii, jj = self.indptr[row:row + 2]
        row_indices = self.indices[ii:jj]
        row_data = self.data[ii:jj]
        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)
        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0
        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])
        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])
        shape = (1, max(0, int(np.ceil(float(stop - start) / stride))))
        return self.__class__((row_data, row_indices, row_indptr), shape=shape, dtype=self.dtype, copy=False)

    def _get_sliceXint(self, row, col):
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        return self._major_slice(row)._get_submatrix(minor=col)

    def _get_sliceXarray(self, row, col):
        return self._major_slice(row)._minor_index_fancy(col)

    def _get_arrayXint(self, row, col):
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    def _get_arrayXslice(self, row, col):
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        return self._major_index_fancy(row)._get_submatrix(minor=col)