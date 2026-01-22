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
class ZipSetupBase:
    ZIP_MODULE = 'data01'

    def setUp(self):
        self.fixtures = contextlib.ExitStack()
        self.addCleanup(self.fixtures.close)
        self.fixtures.enter_context(import_helper.isolated_modules())
        temp_dir = self.fixtures.enter_context(os_helper.temp_dir())
        modules = pathlib.Path(temp_dir) / 'zipped modules.zip'
        src_path = pathlib.Path(__file__).parent.joinpath(self.ZIP_MODULE)
        self.fixtures.enter_context(import_helper.DirsOnSysPath(str(zip_.make_zip_file(src_path, modules))))
        self.data = importlib.import_module(self.ZIP_MODULE)