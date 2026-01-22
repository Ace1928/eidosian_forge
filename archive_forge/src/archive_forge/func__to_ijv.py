from __future__ import annotations
from typing import TYPE_CHECKING
from pandas._libs import lib
from pandas.core.dtypes.missing import notna
from pandas.core.algorithms import factorize
from pandas.core.indexes.api import MultiIndex
from pandas.core.series import Series
def _to_ijv(ss, row_levels: tuple[int] | list[int]=(0,), column_levels: tuple[int] | list[int]=(1,), sort_labels: bool=False) -> tuple[np.ndarray, npt.NDArray[np.intp], npt.NDArray[np.intp], list[IndexLabel], list[IndexLabel]]:
    """
    For an arbitrary MultiIndexed sparse Series return (v, i, j, ilabels,
    jlabels) where (v, (i, j)) is suitable for passing to scipy.sparse.coo
    constructor, and ilabels and jlabels are the row and column labels
    respectively.

    Parameters
    ----------
    ss : Series
    row_levels : tuple/list
    column_levels : tuple/list
    sort_labels : bool, default False
        Sort the row and column labels before forming the sparse matrix.
        When `row_levels` and/or `column_levels` refer to a single level,
        set to `True` for a faster execution.

    Returns
    -------
    values : numpy.ndarray
        Valid values to populate a sparse matrix, extracted from
        ss.
    i_coords : numpy.ndarray (row coordinates of the values)
    j_coords : numpy.ndarray (column coordinates of the values)
    i_labels : list (row labels)
    j_labels : list (column labels)
    """
    _check_is_partition([row_levels, column_levels], range(ss.index.nlevels))
    sp_vals = ss.array.sp_values
    na_mask = notna(sp_vals)
    values = sp_vals[na_mask]
    valid_ilocs = ss.array.sp_index.indices[na_mask]
    i_coords, i_labels = _levels_to_axis(ss, row_levels, valid_ilocs, sort_labels=sort_labels)
    j_coords, j_labels = _levels_to_axis(ss, column_levels, valid_ilocs, sort_labels=sort_labels)
    return (values, i_coords, j_coords, i_labels, j_labels)