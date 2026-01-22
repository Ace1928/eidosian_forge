import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
def _get_format_from_filename(file, mode):
    """Return a format string obtained from file (or file.name).

    If file already exists (= read mode), an empty string is returned on
    error.  If not, an exception is raised.
    The return type will always be str or unicode (even if
    file/file.name is a bytes object).

    """
    format = ''
    file = getattr(file, 'name', file)
    try:
        format = _os.path.splitext(file)[-1][1:]
        format = format.decode('utf-8', 'replace')
    except Exception:
        pass
    if format.upper() not in _formats and 'r' not in mode:
        raise TypeError('No format specified and unable to get format from file extension: {0!r}'.format(file))
    return format