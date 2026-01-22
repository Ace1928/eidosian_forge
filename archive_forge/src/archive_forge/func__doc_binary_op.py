from typing import Iterator, Optional, Tuple
import numpy as np
import pandas
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel
from pandas.util._decorators import doc
from modin.utils import func_from_deprecated_location, hashable
from_pandas = func_from_deprecated_location(
from_arrow = func_from_deprecated_location(
from_dataframe = func_from_deprecated_location(
from_non_pandas = func_from_deprecated_location(
def _doc_binary_op(operation, bin_op, left='Series', right='right', returns='Series'):
    """
    Return callable documenting `Series` or `DataFrame` binary operator.

    Parameters
    ----------
    operation : str
        Operation name.
    bin_op : str
        Binary operation name.
    left : str, default: 'Series'
        The left object to document.
    right : str, default: 'right'
        The right operand name.
    returns : str, default: 'Series'
        Type of returns.

    Returns
    -------
    callable
    """
    if left == 'Series':
        right_type = 'Series or scalar value'
    elif left == 'DataFrame':
        right_type = 'DataFrame, Series or scalar value'
    elif left == 'BasePandasDataset':
        right_type = 'BasePandasDataset or scalar value'
    else:
        raise NotImplementedError(f"Only 'BasePandasDataset', `DataFrame` and 'Series' `left` are allowed, actually passed: {left}")
    doc_op = doc(_doc_binary_operation, operation=operation, right=right, right_type=right_type, bin_op=bin_op, returns=returns, left=left)
    return doc_op