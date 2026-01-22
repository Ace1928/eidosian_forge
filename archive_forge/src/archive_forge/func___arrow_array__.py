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
def __arrow_array__(self, type=None):
    """
        Convert this TensorArray to an ArrowTensorArray extension array.

        This and TensorDtype.__from_arrow__ make up the
        Pandas extension type + array <--> Arrow extension type + array
        interoperability protocol. See
        https://pandas.pydata.org/pandas-docs/stable/development/extending.html#compatibility-with-apache-arrow
        for more information.
        """
    from ray.air.util.tensor_extensions.arrow import ArrowTensorArray, ArrowVariableShapedTensorArray
    if self.is_variable_shaped:
        return ArrowVariableShapedTensorArray.from_numpy(self._tensor)
    else:
        return ArrowTensorArray.from_numpy(self._tensor)