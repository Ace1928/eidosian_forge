from __future__ import annotations
import os
import warnings
from typing import IO, TYPE_CHECKING, Any, Hashable, Iterable, Optional, Union
import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import ArrayLike, Axis, DtypeObj, IndexKeyFunc, Scalar, Sequence
from pandas.api.types import is_integer
from pandas.core.arrays import ExtensionArray
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_dict_like, is_list_like
from pandas.core.series import _coerce_method
from pandas.io.formats.info import SeriesInfo
from pandas.util._validators import validate_bool_kwarg
from modin.config import PersistentPickle
from modin.logging import disable_logging
from modin.pandas.io import from_pandas, to_pandas
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, _inherit_docstrings
from .accessor import CachedAccessor, SparseAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .iterator import PartitionIterator
from .series_utils import (
from .utils import _doc_binary_op, cast_function_modin2pandas, is_scalar
def _create_or_update_from_compiler(self, new_query_compiler, inplace=False) -> Union[Series, None]:
    """
        Return or update a Series with given `new_query_compiler`.

        Parameters
        ----------
        new_query_compiler : PandasQueryCompiler
            QueryCompiler to use to manage the data.
        inplace : bool, default: False
            Whether or not to perform update or creation inplace.

        Returns
        -------
        Series or None
            None if update was done, Series otherwise.
        """
    assert isinstance(new_query_compiler, type(self._query_compiler)) or type(new_query_compiler) in self._query_compiler.__class__.__bases__, 'Invalid Query Compiler object: {}'.format(type(new_query_compiler))
    if not inplace and new_query_compiler.is_series_like():
        return self.__constructor__(query_compiler=new_query_compiler)
    elif not inplace:
        from .dataframe import DataFrame
        return DataFrame(query_compiler=new_query_compiler)
    else:
        self._update_inplace(new_query_compiler=new_query_compiler)