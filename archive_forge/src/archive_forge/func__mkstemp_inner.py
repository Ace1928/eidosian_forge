import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def _mkstemp_inner(dir, pre, suf, flags, output_type):
    """Code common to mkstemp, TemporaryFile, and NamedTemporaryFile."""
    dir = _os.path.abspath(dir)
    names = _get_candidate_names()
    if output_type is bytes:
        names = map(_os.fsencode, names)
    for seq in range(TMP_MAX):
        name = next(names)
        file = _os.path.join(dir, pre + name + suf)
        _sys.audit('tempfile.mkstemp', file)
        try:
            fd = _os.open(file, flags, 384)
        except FileExistsError:
            continue
        except PermissionError:
            if _os.name == 'nt' and _os.path.isdir(dir) and _os.access(dir, _os.W_OK):
                continue
            else:
                raise
        return (fd, file)
    raise FileExistsError(_errno.EEXIST, 'No usable temporary file name found')