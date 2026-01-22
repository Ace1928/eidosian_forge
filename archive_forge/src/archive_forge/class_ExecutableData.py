import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
class ExecutableData(PathData):
    """A :py:class:`PathData` class specifically for executables."""

    @property
    def executable(self):
        """Get (or set) the path to the executable"""
        return self.path()

    @executable.setter
    def executable(self, value):
        self.set_path(value)