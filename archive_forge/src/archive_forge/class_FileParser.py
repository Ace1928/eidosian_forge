import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import pandas as pd
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from triad.collections.dict import ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.io import join, url_to_fs
from triad.utils.pandas_like import PD_UTILS
from fugue.dataframe import LocalBoundedDataFrame, LocalDataFrame, PandasDataFrame
class FileParser(object):

    def __init__(self, path: str, format_hint: Optional[str]=None):
        self._orig_format_hint = format_hint
        self._has_glob = '*' in path or '?' in path
        self._raw_path = path
        self._fs, self._fs_path = url_to_fs(path)
        if not self.is_local:
            self._path = self._fs.unstrip_protocol(self._fs_path)
        else:
            self._path = os.path.abspath(self._fs._strip_protocol(path))
        if format_hint is None or format_hint == '':
            for k, v in _FORMAT_MAP.items():
                if self.suffix.endswith(k):
                    self._format = v
                    return
            raise NotImplementedError(f'{self.suffix} is not supported')
        else:
            assert_or_throw(format_hint in _FORMAT_MAP.values(), NotImplementedError(f'{format_hint} is not supported'))
            self._format = format_hint

    def assert_no_glob(self) -> 'FileParser':
        assert_or_throw(not self.has_glob, f'{self.raw_path} has glob pattern')
        return self

    @property
    def has_glob(self):
        return self._has_glob

    @property
    def is_local(self):
        return isinstance(self._fs, LocalFileSystem)

    def join(self, path: str, format_hint: Optional[str]=None) -> 'FileParser':
        if not self.has_glob:
            _path = join(self.path, path)
        else:
            _path = join(self.parent, path)
        return FileParser(_path, format_hint or self._orig_format_hint)

    @property
    def parent(self) -> str:
        return self._fs.unstrip_protocol(self._fs._parent(self._fs_path))

    @property
    def path(self) -> str:
        return self._path

    @property
    def raw_path(self) -> str:
        return self._raw_path

    @property
    def suffix(self) -> str:
        return ''.join(pathlib.Path(self.raw_path.lower()).suffixes)

    @property
    def file_format(self) -> str:
        return self._format

    def make_parent_dirs(self) -> None:
        self._fs.makedirs(self._fs._parent(self._fs_path), exist_ok=True)

    def find_all(self) -> Iterable['FileParser']:
        if self.has_glob:
            for x in self._fs.glob(self._fs_path):
                yield FileParser(self._fs.unstrip_protocol(x))
        else:
            yield self

    def open(self, *args: Any, **kwargs: Any) -> Any:
        self.assert_no_glob()
        return self._fs.open(self._fs_path, *args, **kwargs)