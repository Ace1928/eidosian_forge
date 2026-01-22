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
@doc(_doc_pandas_parser_class, data_type='JSON files')
class PandasJSONParser(PandasParser):

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop('num_splits', None)
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        if start is not None and end is not None:
            with OpenFile(fname, 'rb', kwargs.pop('compression', 'infer'), **kwargs.pop('storage_options', None) or {}) as bio:
                bio.seek(start)
                to_read = b'' + bio.read(end - start)
            columns = kwargs.pop('columns')
            pandas_df = pandas.read_json(BytesIO(to_read), **kwargs)
        else:
            return pandas.read_json(fname, **kwargs)
        if not pandas_df.columns.equals(columns):
            raise ModinAssumptionError('Columns must be the same across all rows.')
        partition_columns = pandas_df.columns
        return _split_result_for_readers(1, num_splits, pandas_df) + [len(pandas_df), pandas_df.dtypes, partition_columns]