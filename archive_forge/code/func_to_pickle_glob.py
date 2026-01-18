from __future__ import annotations
import inspect
import pathlib
import pickle
from typing import IO, AnyStr, Callable, Iterator, Literal, Optional, Union
import pandas
import pandas._libs.lib as lib
from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, StorageOptions
from modin.core.storage_formats import BaseQueryCompiler
from modin.utils import expanduser_path_arg
from . import DataFrame
@expanduser_path_arg('filepath_or_buffer')
def to_pickle_glob(self, filepath_or_buffer, compression: CompressionOptions='infer', protocol: int=pickle.HIGHEST_PROTOCOL, storage_options: StorageOptions=None) -> None:
    """
    Pickle (serialize) object to file.

    This experimental feature provides parallel writing into multiple pickle files which are
    defined by glob pattern, otherwise (without glob pattern) default pandas implementation is used.

    Parameters
    ----------
    filepath_or_buffer : str
        File path where the pickled object will be stored.
    compression : {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}, default: 'infer'
        A string representing the compression to use in the output file. By
        default, infers from the file extension in specified path.
        Compression mode may be any of the following possible
        values: {{'infer', 'gzip', 'bz2', 'zip', 'xz', None}}. If compression
        mode is 'infer' and path_or_buf is path-like, then detect
        compression mode from the following extensions:
        '.gz', '.bz2', '.zip' or '.xz'. (otherwise no compression).
        If dict given and mode is 'zip' or inferred as 'zip', other entries
        passed as additional compression options.
    protocol : int, default: pickle.HIGHEST_PROTOCOL
        Int which indicates which protocol should be used by the pickler,
        default HIGHEST_PROTOCOL (see `pickle docs <https://docs.python.org/3/library/pickle.html>`_
        paragraph 12.1.2 for details). The possible  values are 0, 1, 2, 3, 4, 5. A negative value
        for the protocol parameter is equivalent to setting its value to HIGHEST_PROTOCOL.
    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc., if using a URL that will be parsed by
        fsspec, e.g., starting "s3://", "gcs://". An error will be raised if providing
        this argument with a non-fsspec URL. See the fsspec and backend storage
        implementation docs for the set of allowed keys and values.
    """
    obj = self
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    if isinstance(self, DataFrame):
        obj = self._query_compiler
    FactoryDispatcher.to_pickle_glob(obj, filepath_or_buffer=filepath_or_buffer, compression=compression, protocol=protocol, storage_options=storage_options)