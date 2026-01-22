from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
def _convert_batch_type_to_numpy(data: DataBatchType) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Convert the provided data to a NumPy ndarray or dict of ndarrays.

    Args:
        data: Data of type DataBatchType

    Returns:
        A numpy representation of the input data.
    """
    pd = _lazy_import_pandas()
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, dict):
        for col_name, col in data.items():
            if not isinstance(col, np.ndarray):
                raise ValueError(f'All values in the provided dict must be of type np.ndarray. Found type {type(col)} for key {col_name} instead.')
        return data
    elif pyarrow is not None and isinstance(data, pyarrow.Table):
        from ray.air.util.tensor_extensions.arrow import ArrowTensorType
        from ray.air.util.transform_pyarrow import _is_column_extension_type, _concatenate_extension_column
        if data.column_names == [TENSOR_COLUMN_NAME] and isinstance(data.schema.types[0], ArrowTensorType):
            return _concatenate_extension_column(data[TENSOR_COLUMN_NAME]).to_numpy(zero_copy_only=False)
        else:
            output_dict = {}
            for col_name in data.column_names:
                col = data[col_name]
                if col.num_chunks == 0:
                    col = pyarrow.array([], type=col.type)
                elif _is_column_extension_type(col):
                    col = _concatenate_extension_column(col)
                else:
                    col = col.combine_chunks()
                output_dict[col_name] = col.to_numpy(zero_copy_only=False)
            return output_dict
    elif isinstance(data, pd.DataFrame):
        return _convert_pandas_to_batch_type(data, BatchFormat.NUMPY)
    else:
        raise ValueError(f'Received data of type: {type(data)}, but expected it to be one of {DataBatchType}')