import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
def _open_code_with_warning(path):
    """Opens the provided file with mode ``'rb'``. This function
    should be used when the intent is to treat the contents as
    executable code.

    ``path`` should be an absolute path.

    When supported by the runtime, this function can be hooked
    in order to allow embedders more control over code files.
    This functionality is not supported on the current runtime.
    """
    import warnings
    warnings.warn('_pyio.open_code() may not be using hooks', RuntimeWarning, 2)
    return open(path, 'rb')