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
def test_useless_loader(self):
    package = create_package(file=FileNotFoundError(), path=FileNotFoundError())
    with self.assertRaises(FileNotFoundError):
        self.execute(package, 'utf-8.file')