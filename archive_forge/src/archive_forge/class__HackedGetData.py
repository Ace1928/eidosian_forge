from _imp import (lock_held, acquire_lock, release_lock,
from importlib._bootstrap import _ERR_MSG, _exec, _load, _builtin_from_name
from importlib._bootstrap_external import SourcelessFileLoader
from importlib import machinery
from importlib import util
import importlib
import os
import sys
import tokenize
import types
import warnings
class _HackedGetData:
    """Compatibility support for 'file' arguments of various load_*()
    functions."""

    def __init__(self, fullname, path, file=None):
        super().__init__(fullname, path)
        self.file = file

    def get_data(self, path):
        """Gross hack to contort loader to deal w/ load_*()'s bad API."""
        if self.file and path == self.path:
            if not self.file.closed:
                file = self.file
                if 'b' not in file.mode:
                    file.close()
            if self.file.closed:
                self.file = file = open(self.path, 'rb')
            with file:
                return file.read()
        else:
            return super().get_data(path)