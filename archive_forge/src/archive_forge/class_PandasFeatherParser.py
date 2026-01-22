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
@doc(_doc_pandas_parser_class, data_type='FEATHER files')
class PandasFeatherParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters='fname : str, path object or file-like object\n    Name of the file, path or file-like object to read.')
    def parse(fname, **kwargs):
        from pyarrow import feather
        num_splits = kwargs.pop('num_splits', None)
        if num_splits is None:
            return pandas.read_feather(fname, **kwargs)
        to_pandas_kwargs = PandasParser.get_types_mapper(kwargs['dtype_backend'])
        del kwargs['dtype_backend']
        with OpenFile(fname, **kwargs.pop('storage_options', None) or {}) as file:
            if not to_pandas_kwargs:
                df = feather.read_feather(file, **kwargs)
            else:
                pa_table = feather.read_table(file, **kwargs)
                df = pa_table.to_pandas(**to_pandas_kwargs)
        return _split_result_for_readers(0, num_splits, df) + [len(df.index), df.dtypes]