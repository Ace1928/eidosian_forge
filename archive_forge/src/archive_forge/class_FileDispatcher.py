import os
import fsspec
import numpy as np
from pandas.io.common import is_fsspec_url, is_url
from modin.config import AsyncReadMode
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
class FileDispatcher(ClassLogger, modin_layer='CORE-IO'):
    """
    Class handles util functions for reading data from different kinds of files.

    Notes
    -----
    `_read`, `deploy`, `parse` and `materialize` are abstract methods and should be
    implemented in the child classes (functions signatures can differ between child
    classes).
    """
    BUFFER_UNSUPPORTED_MSG = 'Reading from buffers or other non-path-like objects is not supported'
    frame_cls = None
    frame_partition_cls = None
    query_compiler_cls = None

    @classmethod
    def read(cls, *args, **kwargs):
        """
        Read data according passed `args` and `kwargs`.

        Parameters
        ----------
        *args : iterable
            Positional arguments to be passed into `_read` function.
        **kwargs : dict
            Keywords arguments to be passed into `_read` function.

        Returns
        -------
        query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `read` is high-level function that calls specific for defined storage format, engine and
        dispatcher class `_read` function with passed parameters and performs some
        postprocessing work on the resulting query_compiler object.
        """
        try:
            query_compiler = cls._read(*args, **kwargs)
        except ModinAssumptionError as err:
            param_name = 'path_or_buf' if 'path_or_buf' in kwargs else 'fname'
            fname = kwargs.pop(param_name)
            return cls.single_worker_read(fname, *args, reason=str(err), **kwargs)
        if not AsyncReadMode.get() and hasattr(query_compiler, 'dtypes'):
            _ = query_compiler.dtypes
        return query_compiler

    @classmethod
    def _read(cls, *args, **kwargs):
        """
        Perform reading of the data from file.

        Should be implemented in the child class.

        Parameters
        ----------
        *args : iterable
            Positional arguments of the function.
        **kwargs : dict
            Keywords arguments of the function.
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def get_path(cls, file_path):
        """
        Process `file_path` in accordance to it's type.

        Parameters
        ----------
        file_path : str, os.PathLike[str] object or file-like object
            The file, or a path to the file. Paths to S3 buckets are also
            acceptable.

        Returns
        -------
        str
            Updated or verified `file_path` parameter.

        Notes
        -----
        if `file_path` is a URL, parameter will be returned as is, otherwise
        absolute path will be returned.
        """
        if is_fsspec_url(file_path) or is_url(file_path):
            return file_path
        else:
            return os.path.abspath(file_path)

    @classmethod
    def file_size(cls, f):
        """
        Get the size of file associated with file handle `f`.

        Parameters
        ----------
        f : file-like object
            File-like object, that should be used to get file size.

        Returns
        -------
        int
            File size in bytes.
        """
        cur_pos = f.tell()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(cur_pos, os.SEEK_SET)
        return size

    @classmethod
    def file_exists(cls, file_path, storage_options=None):
        """
        Check if `file_path` exists.

        Parameters
        ----------
        file_path : str
            String that represents the path to the file (paths to S3 buckets
            are also acceptable).
        storage_options : dict, optional
            Keyword from `read_*` functions.

        Returns
        -------
        bool
            Whether file exists or not.
        """
        if not is_fsspec_url(file_path) and (not is_url(file_path)):
            return os.path.exists(file_path)
        try:
            from botocore.exceptions import ConnectTimeoutError, EndpointConnectionError, NoCredentialsError
            credential_error_type = (NoCredentialsError, PermissionError, EndpointConnectionError, ConnectTimeoutError)
        except ModuleNotFoundError:
            credential_error_type = (PermissionError,)
        if storage_options is not None:
            new_storage_options = dict(storage_options)
            new_storage_options.pop('anon', None)
        else:
            new_storage_options = {}
        fs, _ = fsspec.core.url_to_fs(file_path, **new_storage_options)
        exists = False
        try:
            exists = fs.exists(file_path)
        except credential_error_type:
            fs, _ = fsspec.core.url_to_fs(file_path, anon=True, **new_storage_options)
            exists = fs.exists(file_path)
        return exists

    @classmethod
    def deploy(cls, func, *args, num_returns=1, **kwargs):
        """
        Deploy remote task.

        Should be implemented in the task class (for example in the `RayWrapper`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def parse(self, func, args, num_returns):
        """
        Parse file's data in the worker process.

        Should be implemented in the parser class (for example in the `PandasCSVParser`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def materialize(cls, obj_id):
        """
        Get results from worker.

        Should be implemented in the task class (for example in the `RayWrapper`).
        """
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        row_lengths : list
            Partitions rows lengths.
        column_widths : list
            Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partition objects.
        """
        return np.array([[cls.frame_partition_cls(partition_ids[i][j], length=row_lengths[i], width=column_widths[j]) for j in range(len(partition_ids[i]))] for i in range(len(partition_ids))])

    @classmethod
    def _file_not_found_msg(cls, filename: str):
        return f"No such file: '{filename}'"