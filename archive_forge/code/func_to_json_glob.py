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
@expanduser_path_arg('path_or_buf')
def to_json_glob(self, path_or_buf=None, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=None, storage_options: StorageOptions=None, mode='w') -> None:
    """
    Convert the object to a JSON string.

    Notes
    -----
    * Only string type supported for `path_or_buf` argument.
    * The rest of the arguments are the same as for `pandas.to_json`.
    """
    obj = self
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    if isinstance(self, DataFrame):
        obj = self._query_compiler
    FactoryDispatcher.to_json_glob(obj, path_or_buf=path_or_buf, orient=orient, date_format=date_format, double_precision=double_precision, force_ascii=force_ascii, date_unit=date_unit, default_handler=default_handler, lines=lines, compression=compression, index=index, indent=indent, storage_options=storage_options, mode=mode)