import errno
import os
import shutil
import sys
from .. import tests, ui
from ..clean_tree import clean_tree, iter_deletables
from ..controldir import ControlDir
from ..osutils import supports_symlinks
from . import TestCaseInTempDir
def _dummy_unlink(path):
    """unlink() files other than files named '0foo'.
            """
    if path.endswith('0foo'):
        e = OSError()
        e.errno = errno.EACCES
        raise e