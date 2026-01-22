import ast
from collections import defaultdict
import errno
import functools
import importlib.abc
import importlib.machinery
import importlib.util
import io
import itertools
import marshal
import os
from pathlib import Path
from pathlib import PurePath
import struct
import sys
import tokenize
import types
from typing import Callable
from typing import Dict
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest._io.saferepr import saferepr
from _pytest._version import version
from _pytest.assertion import util
from _pytest.config import Config
from _pytest.main import Session
from _pytest.pathlib import absolutepath
from _pytest.pathlib import fnmatch_ex
from _pytest.stash import StashKey
from _pytest.assertion.util import format_explanation as _format_explanation  # noqa:F401, isort:skip
def _read_pyc(source: Path, pyc: Path, trace: Callable[[str], None]=lambda x: None) -> Optional[types.CodeType]:
    """Possibly read a pytest pyc containing rewritten code.

    Return rewritten code if successful or None if not.
    """
    try:
        fp = open(pyc, 'rb')
    except OSError:
        return None
    with fp:
        try:
            stat_result = os.stat(source)
            mtime = int(stat_result.st_mtime)
            size = stat_result.st_size
            data = fp.read(16)
        except OSError as e:
            trace(f'_read_pyc({source}): OSError {e}')
            return None
        if len(data) != 16:
            trace('_read_pyc(%s): invalid pyc (too short)' % source)
            return None
        if data[:4] != importlib.util.MAGIC_NUMBER:
            trace('_read_pyc(%s): invalid pyc (bad magic number)' % source)
            return None
        if data[4:8] != b'\x00\x00\x00\x00':
            trace('_read_pyc(%s): invalid pyc (unsupported flags)' % source)
            return None
        mtime_data = data[8:12]
        if int.from_bytes(mtime_data, 'little') != mtime & 4294967295:
            trace('_read_pyc(%s): out of date' % source)
            return None
        size_data = data[12:16]
        if int.from_bytes(size_data, 'little') != size & 4294967295:
            trace('_read_pyc(%s): invalid pyc (incorrect size)' % source)
            return None
        try:
            co = marshal.load(fp)
        except Exception as e:
            trace(f'_read_pyc({source}): marshal.load error {e}')
            return None
        if not isinstance(co, types.CodeType):
            trace('_read_pyc(%s): not a code object' % source)
            return None
        return co