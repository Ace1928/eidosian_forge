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
@doc(_doc_pandas_parser_class, data_type='PARQUET data')
class PandasParquetParser(PandasParser):

    @staticmethod
    def _read_row_group_chunk(f, row_group_start, row_group_end, columns, filters, engine, to_pandas_kwargs):
        if engine == 'pyarrow':
            if filters is not None:
                import pyarrow.dataset as ds
                from pyarrow.parquet import filters_to_expression
                parquet_format = ds.ParquetFileFormat()
                fragment = parquet_format.make_fragment(f, row_groups=range(row_group_start, row_group_end))
                dataset = ds.FileSystemDataset([fragment], schema=fragment.physical_schema, format=parquet_format, filesystem=fragment.filesystem)
                metadata = dataset.schema.metadata or {}
                if b'pandas' in metadata and columns is not None:
                    index_columns = json.loads(metadata[b'pandas'].decode('utf8'))['index_columns']
                    index_columns = [col for col in index_columns if not isinstance(col, dict)]
                    columns = list(columns) + list(set(index_columns) - set(columns))
                return dataset.to_table(columns=columns, filter=filters_to_expression(filters)).to_pandas(**to_pandas_kwargs)
            else:
                from pyarrow.parquet import ParquetFile
                return ParquetFile(f).read_row_groups(range(row_group_start, row_group_end), columns=columns, use_pandas_metadata=True).to_pandas(**to_pandas_kwargs)
        elif engine == 'fastparquet':
            from fastparquet import ParquetFile
            return ParquetFile(f)[row_group_start:row_group_end].to_pandas(columns=columns, filters=filters)
        else:
            raise ValueError(f"engine must be one of 'pyarrow', 'fastparquet', got: {engine}")

    @staticmethod
    @doc(_doc_parse_func, parameters='files_for_parser : list\n    List of files to be read.\nengine : str\n    Parquet library to use (either PyArrow or fastparquet).\n')
    def parse(files_for_parser, engine, **kwargs):
        columns = kwargs.get('columns', None)
        filters = kwargs.get('filters', None)
        storage_options = kwargs.get('storage_options', {})
        chunks = []
        if isinstance(files_for_parser, (str, os.PathLike)):
            return pandas.read_parquet(files_for_parser, engine=engine, **kwargs)
        to_pandas_kwargs = PandasParser.get_types_mapper(kwargs['dtype_backend'])
        for file_for_parser in files_for_parser:
            if isinstance(file_for_parser.path, IOBase):
                context = contextlib.nullcontext(file_for_parser.path)
            else:
                context = fsspec.open(file_for_parser.path, **storage_options)
            with context as f:
                chunk = PandasParquetParser._read_row_group_chunk(f, file_for_parser.row_group_start, file_for_parser.row_group_end, columns, filters, engine, to_pandas_kwargs)
            chunks.append(chunk)
        df = pandas.concat(chunks)
        return (df, df.index, len(df))