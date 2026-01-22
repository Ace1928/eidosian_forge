import cupy
import cupyx
from cupyx.scipy import sparse
def _masked_coo(A, mask):
    row = A.row[mask]
    col = A.col[mask]
    data = A.data[mask]
    return sparse.coo_matrix((data, (row, col)), shape=A.shape, dtype=A.dtype)