from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _ndarray_to_column(arr: np.ndarray) -> Union['pd.Series', List[np.ndarray]]:
    """Convert a NumPy ndarray into an appropriate column format for insertion into a
    pandas DataFrame.

    If conversion to a pandas Series fails (e.g. if the ndarray is multi-dimensional),
    fall back to a list of NumPy ndarrays.
    """
    pd = _lazy_import_pandas()
    try:
        return pd.Series(arr)
    except ValueError:
        return list(arr)