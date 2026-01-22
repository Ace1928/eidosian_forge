import errno
import os
import shutil
import sys
from .. import tests, ui
from ..clean_tree import clean_tree, iter_deletables
from ..controldir import ControlDir
from ..osutils import supports_symlinks
from . import TestCaseInTempDir
def _dummy_rmtree(path, ignore_errors=False, onerror=None):
    """Call user supplied error handler onerror.
            """
    try:
        raise OSError
    except OSError as e:
        e.errno = errno.EACCES
        excinfo = sys.exc_info()
        function = os.remove
        if 'subdir0' not in path:
            function = os.listdir
        onerror(function=function, path=path, excinfo=excinfo)