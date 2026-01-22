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
@PublicAPI(stability='beta')
class TensorArrayElement(_TensorOpsMixin, _TensorScalarCastMixin):
    """
    Single element of a TensorArray, wrapping an underlying ndarray.
    """

    def __init__(self, values: np.ndarray):
        """
        Construct a TensorArrayElement from a NumPy ndarray.

        Args:
            values: ndarray that underlies this TensorArray element.
        """
        self._tensor = values

    def __repr__(self):
        return self._tensor.__repr__()

    def __str__(self):
        return self._tensor.__str__()

    @property
    def numpy_dtype(self):
        """
        Get the dtype of the tensor.
        :return: The numpy dtype of the backing ndarray
        """
        return self._tensor.dtype

    @property
    def numpy_ndim(self):
        """
        Get the number of tensor dimensions.
        :return: integer for the number of dimensions
        """
        return self._tensor.ndim

    @property
    def numpy_shape(self):
        """
        Get the shape of the tensor.
        :return: A tuple of integers for the numpy shape of the backing ndarray
        """
        return self._tensor.shape

    @property
    def numpy_size(self):
        """
        Get the size of the tensor.
        :return: integer for the number of elements in the tensor
        """
        return self._tensor.size

    def to_numpy(self):
        """
        Return the values of this element as a NumPy ndarray.
        """
        return np.asarray(self._tensor)

    def __array__(self, dtype: np.dtype=None, **kwargs) -> np.ndarray:
        return np.asarray(self._tensor, dtype=dtype, **kwargs)