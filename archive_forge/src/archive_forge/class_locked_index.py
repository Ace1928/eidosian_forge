import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
class locked_index:
    """Lock the index while making modifications.

    Works as a context manager.
    """

    def __init__(self, path: Union[bytes, str]) -> None:
        self._path = path

    def __enter__(self):
        self._file = GitFile(self._path, 'wb')
        self._index = Index(self._path)
        return self._index

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._file.abort()
            return
        try:
            f = SHA1Writer(self._file)
            write_index_dict(f, self._index._byname)
        except BaseException:
            self._file.abort()
        else:
            f.close()