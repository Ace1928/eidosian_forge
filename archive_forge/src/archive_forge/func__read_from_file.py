from mmap import mmap
import re
import time as _time
from git.compat import defenc
from git.objects.util import (
from git.util import (
import os.path as osp
from typing import Iterator, List, Tuple, Union, TYPE_CHECKING
from git.types import PathLike
def _read_from_file(self) -> None:
    try:
        fmap = file_contents_ro_filepath(self._path, stream=True, allow_mmap=True)
    except OSError:
        return
    try:
        self._deserialize(fmap)
    finally:
        fmap.close()