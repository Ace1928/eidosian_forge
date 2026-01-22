import errno
import filecmp
import os.path
import re
import tempfile
import sys
import subprocess
from collections.abc import MutableSet
def EnsureDirExists(path):
    """Make sure the directory for |path| exists."""
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass