import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List
import numpy as np
import scipy as sp
from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
from ..utils.fixes import pd_fillna
def _split_sparse_columns(arff_data: ArffSparseDataType, include_columns: List) -> ArffSparseDataType:
    """Obtains several columns from sparse ARFF representation. Additionally,
    the column indices are re-labelled, given the columns that are not
    included. (e.g., when including [1, 2, 3], the columns will be relabelled
    to [0, 1, 2]).

    Parameters
    ----------
    arff_data : tuple
        A tuple of three lists of equal size; first list indicating the value,
        second the x coordinate and the third the y coordinate.

    include_columns : list
        A list of columns to include.

    Returns
    -------
    arff_data_new : tuple
        Subset of arff data with only the include columns indicated by the
        include_columns argument.
    """
    arff_data_new: ArffSparseDataType = (list(), list(), list())
    reindexed_columns = {column_idx: array_idx for array_idx, column_idx in enumerate(include_columns)}
    for val, row_idx, col_idx in zip(arff_data[0], arff_data[1], arff_data[2]):
        if col_idx in include_columns:
            arff_data_new[0].append(val)
            arff_data_new[1].append(row_idx)
            arff_data_new[2].append(reindexed_columns[col_idx])
    return arff_data_new