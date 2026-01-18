import sys
import os
from io import StringIO
import textwrap
from distutils.core import Distribution
from distutils.command.build_ext import build_ext
from distutils import sysconfig
from distutils.tests.support import (TempdirManager, LoggingSilencer,
from distutils.extension import Extension
from distutils.errors import (
import unittest
from test import support
from test.support import os_helper
from test.support.script_helper import assert_python_ok
from test.support import threading_helper
@threading_helper.requires_working_threading()
def test_optional_extension(self):
    modules = [Extension('foo', ['xxx'], optional=False)]
    dist = Distribution({'name': 'xx', 'ext_modules': modules})
    cmd = self.build_ext(dist)
    cmd.ensure_finalized()
    self.assertRaises((UnknownFileError, CompileError), cmd.run)
    modules = [Extension('foo', ['xxx'], optional=True)]
    dist = Distribution({'name': 'xx', 'ext_modules': modules})
    cmd = self.build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()