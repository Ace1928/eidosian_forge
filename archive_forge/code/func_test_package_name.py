import abc
import importlib
import io
import sys
import types
import pathlib
import contextlib
from . import data01
from ..abc import ResourceReader
from .compat.py39 import import_helper, os_helper
from . import zip as zip_
from importlib.machinery import ModuleSpec
def test_package_name(self):
    """
        Passing in the package name should succeed.
        """
    self.execute(data01.__name__, 'utf-8.file')