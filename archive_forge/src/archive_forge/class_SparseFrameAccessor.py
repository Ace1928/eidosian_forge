from __future__ import annotations
import pickle
import warnings
from typing import TYPE_CHECKING, Union
import pandas
from pandas._typing import CompressionOptions, StorageOptions
from pandas.core.dtypes.dtypes import SparseDtype
from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.pandas.io import to_dask, to_ray
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.arrays.sparse.accessor.SparseFrameAccessor)
class SparseFrameAccessor(BaseSparseAccessor):

    @classmethod
    def _validate(cls, data: DataFrame):
        """
        Verify that `data` dtypes are compatible with `pandas.core.dtypes.dtypes.SparseDtype`.

        Parameters
        ----------
        data : DataFrame
            Object to check.

        Raises
        ------
        AttributeError
            If check fails.
        """
        dtypes = data.dtypes
        if not all((isinstance(t, SparseDtype) for t in dtypes)):
            raise AttributeError(cls._validation_msg)

    @property
    def density(self):
        return self._parent._default_to_pandas(pandas.DataFrame.sparse).density

    @classmethod
    def from_spmatrix(cls, data, index=None, columns=None):
        ErrorMessage.default_to_pandas('`from_spmatrix`')
        return pd.DataFrame(pandas.DataFrame.sparse.from_spmatrix(data, index=index, columns=columns))

    def to_dense(self):
        return self._default_to_pandas(pandas.DataFrame.sparse.to_dense)

    def to_coo(self):
        return self._default_to_pandas(pandas.DataFrame.sparse.to_coo)