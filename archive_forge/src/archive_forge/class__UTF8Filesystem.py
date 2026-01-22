import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class _UTF8Filesystem(Feature):
    """Is the filesystem UTF-8?"""

    def _probe(self):
        if sys.getfilesystemencoding().upper() in ('UTF-8', 'UTF8'):
            return True
        return False