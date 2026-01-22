from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _convert_batch_type_to_pandas(data: DataBatchType, cast_tensor_columns: bool=False) -> 'pd.DataFrame':
    """Convert the provided data to a Pandas DataFrame.

    Args:
        data: Data of type DataBatchType
        cast_tensor_columns: Whether tensor columns should be cast to NumPy ndarrays.

    Returns:
        A pandas Dataframe representation of the input data.

    """
    pd = _lazy_import_pandas()
    if isinstance(data, np.ndarray):
        data = pd.DataFrame({TENSOR_COLUMN_NAME: _ndarray_to_column(data)})
    elif isinstance(data, dict):
        tensor_dict = {}
        for col_name, col in data.items():
            if not isinstance(col, np.ndarray):
                raise ValueError(f'All values in the provided dict must be of type np.ndarray. Found type {type(col)} for key {col_name} instead.')
            tensor_dict[col_name] = _ndarray_to_column(col)
        data = pd.DataFrame(tensor_dict)
    elif pyarrow is not None and isinstance(data, pyarrow.Table):
        data = data.to_pandas()
    elif not isinstance(data, pd.DataFrame):
        raise ValueError(f'Received data of type: {type(data)}, but expected it to be one of {DataBatchType}')
    if cast_tensor_columns:
        data = _cast_tensor_columns_to_ndarrays(data)
    return data