import importlib
import os
import stat
import subprocess
import sys
import tempfile
import warnings
from .. import osutils, symbol_versioning
class Win32Feature(Feature):
    """Feature testing whether we're running selftest on Windows
    or Windows-like platform.
    """

    def _probe(self):
        return sys.platform == 'win32'

    def feature_name(self):
        return 'win32 platform'