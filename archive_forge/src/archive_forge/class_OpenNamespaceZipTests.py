import textwrap
import unittest
import warnings
import importlib
import contextlib
import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper
class OpenNamespaceZipTests(FilesTests, util.ZipSetup, unittest.TestCase):
    ZIP_MODULE = 'namespacedata01'