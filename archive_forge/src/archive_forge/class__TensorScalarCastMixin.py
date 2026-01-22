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
class _TensorScalarCastMixin:
    """
    Mixin for casting scalar tensors to a particular numeric type.
    """

    def _scalarfunc(self, func: Callable[[Any], Any]):
        return func(self._tensor)

    def __complex__(self):
        return self._scalarfunc(complex)

    def __float__(self):
        return self._scalarfunc(float)

    def __int__(self):
        return self._scalarfunc(int)

    def __hex__(self):
        return self._scalarfunc(hex)

    def __oct__(self):
        return self._scalarfunc(oct)