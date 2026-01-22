import atexit
import errno
import importlib
import os
import sys
import tempfile
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
def _remove_file(file_name):
    """Remove a file, if it exists."""
    try:
        os.remove(file_name)
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise