import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def _bmat(list_obj):
    """
    Helper function to create a block matrix in cupy from a list
    of smaller 2D dense arrays
    """
    n_rows = len(list_obj)
    n_cols = len(list_obj[0])
    final_shape = [0, 0]
    for i in range(n_rows):
        final_shape[0] += list_obj[i][0].shape[0]
    for j in range(n_cols):
        final_shape[1] += list_obj[0][j].shape[1]
    dtype = cupy.result_type(*[arr.dtype for list_iter in list_obj for arr in list_iter])
    F_order = all((arr.flags['F_CONTIGUOUS'] for list_iter in list_obj for arr in list_iter))
    C_order = all((arr.flags['C_CONTIGUOUS'] for list_iter in list_obj for arr in list_iter))
    order = 'F' if F_order and (not C_order) else 'C'
    result = cupy.empty(tuple(final_shape), dtype=dtype, order=order)
    start_idx_row = 0
    start_idx_col = 0
    end_idx_row = 0
    end_idx_col = 0
    for i in range(n_rows):
        end_idx_row = start_idx_row + list_obj[i][0].shape[0]
        start_idx_col = 0
        for j in range(n_cols):
            end_idx_col = start_idx_col + list_obj[i][j].shape[1]
            result[start_idx_row:end_idx_row, start_idx_col:end_idx_col] = list_obj[i][j]
            start_idx_col = end_idx_col
        start_idx_row = end_idx_row
    return result