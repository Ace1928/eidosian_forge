from __future__ import annotations
import io
from types import ModuleType
from typing import (
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.compat._optional import import_optional_dependency
from pandas.util._validators import check_dtype_backend
import pandas as pd
from pandas.core.indexes.api import default_index
from pandas.io._util import arrow_string_types_mapper
from pandas.io.common import (

    Write a DataFrame to the ORC format.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    df : DataFrame
        The dataframe to be written to ORC. Raises NotImplementedError
        if dtype of one or more columns is category, unsigned integers,
        intervals, periods or sparse.
    path : str, file-like object or None, default None
        If a string, it will be used as Root Directory path
        when writing a partitioned dataset. By file-like object,
        we refer to objects with a write() method, such as a file handle
        (e.g. via builtin open function). If path is None,
        a bytes object is returned.
    engine : str, default 'pyarrow'
        ORC library to use.
    index : bool, optional
        If ``True``, include the dataframe's index(es) in the file output. If
        ``False``, they will not be written to the file.
        If ``None``, similar to ``infer`` the dataframe's index(es)
        will be saved. However, instead of being saved as values,
        the RangeIndex will be stored as a range in the metadata so it
        doesn't require much space and is faster. Other indexes will
        be included as columns in the file output.
    engine_kwargs : dict[str, Any] or None, default None
        Additional keyword arguments passed to :func:`pyarrow.orc.write_table`.

    Returns
    -------
    bytes if no path argument is provided else None

    Raises
    ------
    NotImplementedError
        Dtype of one or more columns is category, unsigned integers, interval,
        period or sparse.
    ValueError
        engine is not pyarrow.

    Notes
    -----
    * Before using this function you should read the
      :ref:`user guide about ORC <io.orc>` and
      :ref:`install optional dependencies <install.warn_orc>`.
    * This function requires `pyarrow <https://arrow.apache.org/docs/python/>`_
      library.
    * For supported dtypes please refer to `supported ORC features in Arrow
      <https://arrow.apache.org/docs/cpp/orc.html#data-types>`__.
    * Currently timezones in datetime columns are not preserved when a
      dataframe is converted into ORC files.
    