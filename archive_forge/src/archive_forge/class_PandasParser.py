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
class PandasParser(ClassLogger, modin_layer='PARSER'):
    """Base class for parser classes with pandas storage format."""

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def generic_parse(fname, **kwargs):
        warnings.filterwarnings('ignore')
        num_splits = kwargs.pop('num_splits', None)
        start = kwargs.pop('start', None)
        end = kwargs.pop('end', None)
        header_size = kwargs.pop('header_size', 0)
        common_dtypes = kwargs.pop('common_dtypes', None)
        encoding = kwargs.get('encoding', None)
        callback = kwargs.pop('callback')
        if start is None or end is None:
            return callback(fname, **kwargs)
        with OpenFile(fname, 'rb', kwargs.pop('compression', 'infer'), **kwargs.pop('storage_options', None) or {}) as bio:
            header = b''
            if encoding and ('utf' in encoding and '8' not in encoding or encoding == 'unicode_escape' or encoding.replace('-', '_') == 'utf_8_sig'):
                fio = TextIOWrapper(bio, encoding=encoding, newline='')
                if header_size == 0:
                    header = fio.readline().encode(encoding)
                    kwargs['skiprows'] = 1
                for _ in range(header_size):
                    header += fio.readline().encode(encoding)
            elif encoding is not None:
                if header_size == 0:
                    header = bio.readline()
                    kwargs['skiprows'] = 1
                for _ in range(header_size):
                    header += bio.readline()
            else:
                for _ in range(header_size):
                    header += bio.readline()
            bio.seek(start)
            to_read = header + bio.read(end - start)
        if 'memory_map' in kwargs:
            kwargs = kwargs.copy()
            del kwargs['memory_map']
        if common_dtypes is not None:
            kwargs['dtype'] = common_dtypes
        pandas_df = callback(BytesIO(to_read), **kwargs)
        index = pandas_df.index if not isinstance(pandas_df.index, pandas.RangeIndex) else len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [index, pandas_df.dtypes]

    @classmethod
    def get_dtypes(cls, dtypes_ids, columns):
        """
        Get common for all partitions dtype for each of the columns.

        Parameters
        ----------
        dtypes_ids : list
            Array with references to the partitions dtypes objects.
        columns : array-like or Index (1d)
            The names of the columns in this variable will be used
            for dtypes creation.

        Returns
        -------
        frame_dtypes : pandas.Series, dtype or None
            Resulting dtype or pandas.Series where column names are used as
            index and types of columns are used as values for full resulting
            frame.
        """
        if len(dtypes_ids) == 0:
            return None
        partitions_dtypes = cls.materialize(dtypes_ids)
        if all([len(dtype) == 0 for dtype in partitions_dtypes]):
            return None
        combined_part_dtypes = pandas.concat(partitions_dtypes, axis=1)
        frame_dtypes = combined_part_dtypes.iloc[:, 0]
        frame_dtypes.name = None
        if not combined_part_dtypes.eq(frame_dtypes, axis=0).all(axis=None):
            ErrorMessage.mismatch_with_pandas(operation='read_*', message='Data types of partitions are different! ' + 'Please refer to the troubleshooting section of the Modin documentation ' + 'to fix this issue')
            frame_dtypes = combined_part_dtypes.apply(lambda row: find_common_type_cat(row.values), axis=1).squeeze(axis=0)
        if isinstance(frame_dtypes, pandas.Series):
            frame_dtypes.index = columns
        else:
            frame_dtypes = pandas.Series(frame_dtypes, index=columns)
        return frame_dtypes

    @classmethod
    def single_worker_read(cls, fname, *args, reason: str, **kwargs):
        """
        Perform reading by single worker (default-to-pandas implementation).

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file or file-like object to read.
        *args : tuple
            Positional arguments to be passed into `read_*` function.
        reason : str
            Message describing the reason for falling back to pandas.
        **kwargs : dict
            Keywords arguments to be passed into `read_*` function.

        Returns
        -------
        BaseQueryCompiler or
        dict or
        pandas.io.parsers.TextFileReader
            Object with imported data (or with reference to data) for further
            processing, object type depends on the child class `parse` function
            result type.
        """
        ErrorMessage.default_to_pandas(reason=reason)
        pandas_frame = cls.parse(fname, *args, **kwargs)
        if isinstance(pandas_frame, pandas.io.parsers.TextFileReader):
            pd_read = pandas_frame.read
            pandas_frame.read = lambda *args, **kwargs: cls.query_compiler_cls.from_pandas(pd_read(*args, **kwargs), cls.frame_cls)
            return pandas_frame
        elif isinstance(pandas_frame, dict):
            return {i: cls.query_compiler_cls.from_pandas(frame, cls.frame_cls) for i, frame in pandas_frame.items()}
        return cls.query_compiler_cls.from_pandas(pandas_frame, cls.frame_cls)

    @staticmethod
    def get_types_mapper(dtype_backend):
        """
        Get types mapper that would be used in read_parquet/read_feather.

        Parameters
        ----------
        dtype_backend : {"numpy_nullable", "pyarrow", lib.no_default}

        Returns
        -------
        dict
        """
        to_pandas_kwargs = {}
        if dtype_backend == 'numpy_nullable':
            from pandas.io._util import _arrow_dtype_mapping
            mapping = _arrow_dtype_mapping()
            to_pandas_kwargs['types_mapper'] = mapping.get
        elif dtype_backend == 'pyarrow':
            to_pandas_kwargs['types_mapper'] = pandas.ArrowDtype
        return to_pandas_kwargs
    infer_compression = infer_compression