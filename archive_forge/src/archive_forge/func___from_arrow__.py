import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
def __from_arrow__(self, array: Union[pa.Array, pa.ChunkedArray]):
    """
        Convert a pyarrow (chunked) array to a TensorArray.

        This and TensorArray.__arrow_array__ make up the
        Pandas extension type + array <--> Arrow extension type + array
        interoperability protocol. See
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#compatibility-with-apache-arrow
        for more information.
        """
    if isinstance(array, pa.ChunkedArray):
        if array.num_chunks > 1:
            values = np.concatenate([chunk.to_numpy() for chunk in array.iterchunks()])
        else:
            values = array.chunk(0).to_numpy()
    else:
        values = array.to_numpy()
    return TensorArray(values)