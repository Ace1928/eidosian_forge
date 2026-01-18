import numpy
import cupy
from cupy import _core
from cupyx import cusparse
from cupyx.scipy.sparse import _base
from cupyx.scipy.sparse import _csc
from cupyx.scipy.sparse import _csr
from cupyx.scipy.sparse import _data as sparse_data
from cupyx.scipy.sparse import _util
from cupyx.scipy.sparse import _sputils
def tocsr(self, copy=False):
    """Converts the matrix to Compressed Sparse Row format.

        Args:
            copy (bool): If ``False``, it shares data arrays as much as
                possible. Actually this option is ignored because all
                arrays in a matrix cannot be shared in coo to csr conversion.

        Returns:
            cupyx.scipy.sparse.csr_matrix: Converted matrix.

        """
    if self.nnz == 0:
        return _csr.csr_matrix(self.shape, dtype=self.dtype)
    x = self.copy()
    x.sum_duplicates()
    cusparse.coosort(x, 'r')
    x = cusparse.coo2csr(x)
    x.has_canonical_format = True
    return x