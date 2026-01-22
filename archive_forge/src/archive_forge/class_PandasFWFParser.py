import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple
import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
from modin.config import MinPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@doc(_doc_pandas_parser_class, data_type='tables with fixed-width formatted lines')
class PandasFWFParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common2)
    def parse(fname, common_read_kwargs, **kwargs):
        return PandasParser.generic_parse(fname, callback=PandasFWFParser.read_callback, **common_read_kwargs, **kwargs)

    @staticmethod
    def read_callback(*args, **kwargs):
        """
        Parse data on each partition.

        Parameters
        ----------
        *args : list
            Positional arguments to be passed to the callback function.
        **kwargs : dict
            Keyword arguments to be passed to the callback function.

        Returns
        -------
        pandas.DataFrame or pandas.io.parsers.TextFileReader
            Function call result.
        """
        return pandas.read_fwf(*args, **kwargs)