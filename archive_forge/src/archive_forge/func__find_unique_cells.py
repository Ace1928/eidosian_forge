from . import measure
from . import select
from . import utils
from scipy import sparse
import numbers
import numpy as np
import pandas as pd
import warnings
def _find_unique_cells(data):
    """Identify unique cells.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    unique_idx : np.ndarray
        Sorted array of unique element indices
    """
    if utils.is_SparseDataFrame(data):
        unique_idx = _find_unique_cells(data.to_coo())
    elif utils.is_sparse_dataframe(data):
        unique_idx = _find_unique_cells(data.sparse.to_coo())
    elif isinstance(data, pd.DataFrame):
        unique_idx = ~data.duplicated()
    elif isinstance(data, np.ndarray):
        _, unique_idx = np.unique(data, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
    elif sparse.issparse(data):
        _, unique_data = np.unique(data.tolil().data, return_index=True)
        _, unique_index = np.unique(data.tolil().rows, return_index=True)
        unique_idx = np.sort(list(set(unique_index).union(set(unique_data))))
    return unique_idx