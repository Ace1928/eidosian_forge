import json
import os
import re
from typing import TYPE_CHECKING
import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
class ColumnStoreDataset:
    """
    Base class that encapsulates Parquet engine-specific details.

    This class exposes a set of functions that are commonly used in the
    `read_parquet` implementation.

    Attributes
    ----------
    path : str, path object or file-like object
        The filepath of the parquet file in local filesystem or hdfs.
    storage_options : dict
        Parameters for specific storage engine.
    _fs_path : str, path object or file-like object
        The filepath or handle of the parquet dataset specific to the
        filesystem implementation. E.g. for `s3://test/example`, _fs
        would be set to S3FileSystem and _fs_path would be `test/example`.
    _fs : Filesystem
        Filesystem object specific to the given parquet file/dataset.
    dataset : ParquetDataset or ParquetFile
        Underlying dataset implementation for PyArrow and fastparquet
        respectively.
    _row_groups_per_file : list
        List that contains the number of row groups for each file in the
        given parquet dataset.
    _files : list
        List that contains the full paths of the parquet files in the dataset.
    """

    def __init__(self, path, storage_options):
        self.path = path.__fspath__() if isinstance(path, os.PathLike) else path
        self.storage_options = storage_options
        self._fs_path = None
        self._fs = None
        self.dataset = self._init_dataset()
        self._row_groups_per_file = None
        self._files = None

    @property
    def pandas_metadata(self):
        """Return the pandas metadata of the dataset."""
        raise NotImplementedError

    @property
    def columns(self):
        """Return the list of columns in the dataset."""
        raise NotImplementedError

    @property
    def engine(self):
        """Return string representing what engine is being used."""
        raise NotImplementedError

    @property
    def files(self):
        """Return the list of formatted file paths of the dataset."""
        raise NotImplementedError

    @property
    def row_groups_per_file(self):
        """Return a list with the number of row groups per file."""
        raise NotImplementedError

    @property
    def fs(self):
        """
        Return the filesystem object associated with the dataset path.

        Returns
        -------
        filesystem
            Filesystem object.
        """
        if self._fs is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs = self.path.fs
            else:
                self._fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        return self._fs

    @property
    def fs_path(self):
        """
        Return the filesystem-specific path or file handle.

        Returns
        -------
        fs_path : str, path object or file-like object
            String path specific to filesystem or a file handle.
        """
        if self._fs_path is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs_path = self.path
            else:
                self._fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        return self._fs_path

    def to_pandas_dataframe(self, columns):
        """
        Read the given columns as a pandas dataframe.

        Parameters
        ----------
        columns : list
            List of columns that should be read from file.
        """
        raise NotImplementedError

    def _get_files(self, files):
        """
        Retrieve list of formatted file names in dataset path.

        Parameters
        ----------
        files : list
            List of files from path.

        Returns
        -------
        fs_files : list
            List of files from path with fs-protocol prepended.
        """

        def _unstrip_protocol(protocol, path):
            protos = (protocol,) if isinstance(protocol, str) else protocol
            for protocol in protos:
                if path.startswith(f'{protocol}://'):
                    return path
            return f'{protos[0]}://{path}'
        if isinstance(self.path, AbstractBufferedFile):
            return [self.path]
        if version.parse(fsspec.__version__) < version.parse('2022.5.0'):
            fs_files = [_unstrip_protocol(self.fs.protocol, fpath) for fpath in files]
        else:
            fs_files = [self.fs.unstrip_protocol(fpath) for fpath in files]
        return fs_files