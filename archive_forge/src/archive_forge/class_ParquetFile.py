from collections import defaultdict
from contextlib import nullcontext
from functools import reduce
import inspect
import json
import os
import re
import operator
import warnings
import pyarrow as pa
from pyarrow._parquet import (ParquetReader, Statistics,  # noqa
from pyarrow.fs import (LocalFileSystem, FileSystem, FileType,
from pyarrow import filesystem as legacyfs
from pyarrow.util import guid, _is_path_like, _stringify_path, _deprecate_api
class ParquetFile:
    """
    Reader interface for a single Parquet file.

    Parameters
    ----------
    source : str, pathlib.Path, pyarrow.NativeFile, or file-like object
        Readable source. For passing bytes or buffer-like file containing a
        Parquet file, use pyarrow.BufferReader.
    metadata : FileMetaData, default None
        Use existing metadata object, rather than reading from file.
    common_metadata : FileMetaData, default None
        Will be used in reads for pandas schema metadata if not found in the
        main file's metadata, no other uses at the moment.
    read_dictionary : list
        List of column names to read directly as DictionaryArray.
    memory_map : bool, default False
        If the source is a file path, use a memory map to read file, which can
        improve performance in some environments.
    buffer_size : int, default 0
        If positive, perform read buffering when deserializing individual
        column chunks. Otherwise IO calls are unbuffered.
    pre_buffer : bool, default False
        Coalesce and issue file reads in parallel to improve performance on
        high-latency filesystems (e.g. S3). If True, Arrow will use a
        background I/O thread pool.
    coerce_int96_timestamp_unit : str, default None
        Cast timestamps that are stored in INT96 format to a particular
        resolution (e.g. 'ms'). Setting to None is equivalent to 'ns'
        and therefore INT96 timestamps will be inferred as timestamps
        in nanoseconds.
    decryption_properties : FileDecryptionProperties, default None
        File decryption properties for Parquet Modular Encryption.
    thrift_string_size_limit : int, default None
        If not None, override the maximum total string size allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    thrift_container_size_limit : int, default None
        If not None, override the maximum total size of containers allocated
        when decoding Thrift structures. The default limit should be
        sufficient for most Parquet files.
    filesystem : FileSystem, default None
        If nothing passed, will be inferred based on path.
        Path will try to be found in the local on-disk filesystem otherwise
        it will be parsed as an URI to determine the filesystem.
    page_checksum_verification : bool, default False
        If True, verify the checksum for each page read from the file.

    Examples
    --------

    Generate an example PyArrow Table and write it to Parquet file:

    >>> import pyarrow as pa
    >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
    ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
    ...                              "Brittle stars", "Centipede"]})

    >>> import pyarrow.parquet as pq
    >>> pq.write_table(table, 'example.parquet')

    Create a ``ParquetFile`` object from the Parquet file:

    >>> parquet_file = pq.ParquetFile('example.parquet')

    Read the data:

    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: string
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [["Flamingo","Parrot","Dog","Horse","Brittle stars","Centipede"]]

    Create a ParquetFile object with "animal" column as DictionaryArray:

    >>> parquet_file = pq.ParquetFile('example.parquet',
    ...                               read_dictionary=["animal"])
    >>> parquet_file.read()
    pyarrow.Table
    n_legs: int64
    animal: dictionary<values=string, indices=int32, ordered=0>
    ----
    n_legs: [[2,2,4,4,5,100]]
    animal: [  -- dictionary:
    ["Flamingo","Parrot",...,"Brittle stars","Centipede"]  -- indices:
    [0,1,2,3,4,5]]
    """

    def __init__(self, source, *, metadata=None, common_metadata=None, read_dictionary=None, memory_map=False, buffer_size=0, pre_buffer=False, coerce_int96_timestamp_unit=None, decryption_properties=None, thrift_string_size_limit=None, thrift_container_size_limit=None, filesystem=None, page_checksum_verification=False):
        self._close_source = getattr(source, 'closed', True)
        filesystem, source = _resolve_filesystem_and_path(source, filesystem, memory_map)
        if filesystem is not None:
            source = filesystem.open_input_file(source)
            self._close_source = True
        self.reader = ParquetReader()
        self.reader.open(source, use_memory_map=memory_map, buffer_size=buffer_size, pre_buffer=pre_buffer, read_dictionary=read_dictionary, metadata=metadata, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit, decryption_properties=decryption_properties, thrift_string_size_limit=thrift_string_size_limit, thrift_container_size_limit=thrift_container_size_limit, page_checksum_verification=page_checksum_verification)
        self.common_metadata = common_metadata
        self._nested_paths_by_prefix = self._build_nested_paths()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def _build_nested_paths(self):
        paths = self.reader.column_paths
        result = defaultdict(list)
        for i, path in enumerate(paths):
            key = path[0]
            rest = path[1:]
            while True:
                result[key].append(i)
                if not rest:
                    break
                key = '.'.join((key, rest[0]))
                rest = rest[1:]
        return result

    @property
    def metadata(self):
        """
        Return the Parquet metadata.
        """
        return self.reader.metadata

    @property
    def schema(self):
        """
        Return the Parquet schema, unconverted to Arrow types
        """
        return self.metadata.schema

    @property
    def schema_arrow(self):
        """
        Return the inferred Arrow schema, converted from the whole Parquet
        file's schema

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read the Arrow schema:

        >>> parquet_file.schema_arrow
        n_legs: int64
        animal: string
        """
        return self.reader.schema_arrow

    @property
    def num_row_groups(self):
        """
        Return the number of row groups of the Parquet file.

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.num_row_groups
        1
        """
        return self.reader.num_row_groups

    def close(self, force: bool=False):
        if self._close_source or force:
            self.reader.close()

    @property
    def closed(self) -> bool:
        return self.reader.closed

    def read_row_group(self, i, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read a single row group from a Parquet file.

        Parameters
        ----------
        i : int
            Index of the individual row group that we want to read.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row group as a table (of columns)

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_group(0)
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,100]]
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_row_group(i, column_indices=column_indices, use_threads=use_threads)

    def read_row_groups(self, row_groups, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read a multiple row groups from a Parquet file.

        Parameters
        ----------
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the row group. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the row groups as a table (of columns).

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.read_row_groups([0,0])
        pyarrow.Table
        n_legs: int64
        animal: string
        ----
        n_legs: [[2,2,4,4,5,...,2,4,4,5,100]]
        animal: [["Flamingo","Parrot","Dog",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_row_groups(row_groups, column_indices=column_indices, use_threads=use_threads)

    def iter_batches(self, batch_size=65536, row_groups=None, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read streaming batches from a Parquet file.

        Parameters
        ----------
        batch_size : int, default 64K
            Maximum number of records to yield per batch. Batches may be
            smaller if there aren't enough rows in the file.
        row_groups : list
            Only these row groups will be read from the file.
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : boolean, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : boolean, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Yields
        ------
        pyarrow.RecordBatch
            Contents of each batch as a record batch

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')
        >>> for i in parquet_file.iter_batches():
        ...     print("RecordBatch")
        ...     print(i.to_pandas())
        ...
        RecordBatch
           n_legs         animal
        0       2       Flamingo
        1       2         Parrot
        2       4            Dog
        3       4          Horse
        4       5  Brittle stars
        5     100      Centipede
        """
        if row_groups is None:
            row_groups = range(0, self.metadata.num_row_groups)
        column_indices = self._get_column_indices(columns, use_pandas_metadata=use_pandas_metadata)
        batches = self.reader.iter_batches(batch_size, row_groups=row_groups, column_indices=column_indices, use_threads=use_threads)
        return batches

    def read(self, columns=None, use_threads=True, use_pandas_metadata=False):
        """
        Read a Table from Parquet format.

        Parameters
        ----------
        columns : list
            If not None, only these columns will be read from the file. A
            column name may be a prefix of a nested field, e.g. 'a' will select
            'a.b', 'a.c', and 'a.d.e'.
        use_threads : bool, default True
            Perform multi-threaded column reads.
        use_pandas_metadata : bool, default False
            If True and file has custom pandas schema metadata, ensure that
            index columns are also loaded.

        Returns
        -------
        pyarrow.table.Table
            Content of the file as a table (of columns).

        Examples
        --------
        Generate an example Parquet file:

        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        Read a Table:

        >>> parquet_file.read(columns=["animal"])
        pyarrow.Table
        animal: string
        ----
        animal: [["Flamingo","Parrot",...,"Brittle stars","Centipede"]]
        """
        column_indices = self._get_column_indices(columns, use_pandas_metadata=use_pandas_metadata)
        return self.reader.read_all(column_indices=column_indices, use_threads=use_threads)

    def scan_contents(self, columns=None, batch_size=65536):
        """
        Read contents of file for the given columns and batch size.

        Notes
        -----
        This function's primary purpose is benchmarking.
        The scan is executed on a single thread.

        Parameters
        ----------
        columns : list of integers, default None
            Select columns to read, if None scan all columns.
        batch_size : int, default 64K
            Number of rows to read at a time internally.

        Returns
        -------
        num_rows : int
            Number of rows in file

        Examples
        --------
        >>> import pyarrow as pa
        >>> table = pa.table({'n_legs': [2, 2, 4, 4, 5, 100],
        ...                   'animal': ["Flamingo", "Parrot", "Dog", "Horse",
        ...                              "Brittle stars", "Centipede"]})
        >>> import pyarrow.parquet as pq
        >>> pq.write_table(table, 'example.parquet')
        >>> parquet_file = pq.ParquetFile('example.parquet')

        >>> parquet_file.scan_contents()
        6
        """
        column_indices = self._get_column_indices(columns)
        return self.reader.scan_contents(column_indices, batch_size=batch_size)

    def _get_column_indices(self, column_names, use_pandas_metadata=False):
        if column_names is None:
            return None
        indices = []
        for name in column_names:
            if name in self._nested_paths_by_prefix:
                indices.extend(self._nested_paths_by_prefix[name])
        if use_pandas_metadata:
            file_keyvalues = self.metadata.metadata
            common_keyvalues = self.common_metadata.metadata if self.common_metadata is not None else None
            if file_keyvalues and b'pandas' in file_keyvalues:
                index_columns = _get_pandas_index_columns(file_keyvalues)
            elif common_keyvalues and b'pandas' in common_keyvalues:
                index_columns = _get_pandas_index_columns(common_keyvalues)
            else:
                index_columns = []
            if indices is not None and index_columns:
                indices += [self.reader.column_name_idx(descr) for descr in index_columns if not isinstance(descr, dict)]
        return indices