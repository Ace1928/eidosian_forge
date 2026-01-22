import functools
import inspect
import os
from csv import Dialect
from typing import Callable, Dict, Sequence, Tuple, Union
import pandas
import pandas._libs.lib as lib
import pyarrow as pa
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import get_handle, is_url
from pyarrow.csv import ConvertOptions, ParseOptions, ReadOptions, read_csv
from modin.core.io import BaseIO
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.error_message import ErrorMessage
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.dataframe import (
from modin.experimental.core.storage_formats.hdk.query_compiler import (
from modin.utils import _inherit_docstrings
@classmethod
def _prepare_pyarrow_usecols(cls, read_csv_kwargs):
    """
        Define `usecols` parameter in the way PyArrow can process it.

        Parameters
        ----------
        read_csv_kwargs : dict
            Parameters of read_csv.

        Returns
        -------
        list
            Redefined `usecols` parameter.
        """
    usecols = read_csv_kwargs['usecols']
    engine = read_csv_kwargs['engine']
    usecols_md, usecols_names_dtypes = cls._validate_usecols_arg(usecols)
    if usecols_md:
        empty_pd_df = pandas.read_csv(**dict(read_csv_kwargs, nrows=0, skipfooter=0, usecols=None, engine=None if engine == 'arrow' else engine))
        column_names = empty_pd_df.columns
        if usecols_names_dtypes == 'string':
            if usecols_md.issubset(set(column_names)):
                usecols_md = [col_name for col_name in column_names if col_name in usecols_md]
            else:
                raise NotImplementedError("values passed in the `usecols` parameter don't match columns names")
        elif usecols_names_dtypes == 'integer':
            usecols_md = sorted(usecols_md)
            if len(column_names) < usecols_md[-1]:
                raise NotImplementedError('max usecols value is higher than the number of columns')
            usecols_md = [column_names[i] for i in usecols_md]
        elif callable(usecols_md):
            usecols_md = [col_name for col_name in column_names if usecols_md(col_name)]
        else:
            raise NotImplementedError('unsupported `usecols` parameter')
    return usecols_md