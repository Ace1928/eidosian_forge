from types import FunctionType
from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot
def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc, normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i * cols:(i + 1) * cols], mat[j * cols:(j + 1) * cols] = (mat[j * cols:(j + 1) * cols], mat[i * cols:(i + 1) * cols])

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i) * cols
        for p in range(i * cols, (i + 1) * cols):
            mat[p] = isimp(a * mat[p] - b * mat[p + q])
    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row, piv_col = (0, 0)
    pivot_cols = []
    swaps = []
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val, assumed_nonzero, newly_determined = _find_reasonable_pivot(get_col(piv_col)[piv_row:], iszerofunc, simpfunc)
        for offset, val in newly_determined:
            offset += piv_row
            mat[offset * cols + piv_col] = val
        if pivot_offset is None:
            piv_col += 1
            continue
        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))
        if normalize_last is False:
            i, j = (piv_row, piv_col)
            mat[i * cols + j] = one
            for p in range(i * cols + j + 1, (i + 1) * cols):
                mat[p] = isimp(mat[p] / pivot_val)
            pivot_val = one
        for row in range(rows):
            if row == piv_row:
                continue
            if zero_above is False and row < piv_row:
                continue
            val = mat[row * cols + piv_col]
            if iszerofunc(val):
                continue
            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i * cols + piv_j]
            mat[piv_i * cols + piv_j] = one
            for p in range(piv_i * cols + piv_j + 1, (piv_i + 1) * cols):
                mat[p] = isimp(mat[p] / pivot_val)
    return (mat, tuple(pivot_cols), tuple(swaps))