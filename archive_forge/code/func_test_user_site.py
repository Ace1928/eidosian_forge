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
def test_user_site(self):
    import site
    dist = Distribution({'name': 'xx'})
    cmd = self.build_ext(dist)
    options = [name for name, short, lable in cmd.user_options]
    self.assertIn('user', options)
    cmd.user = 1
    lib = os.path.join(site.USER_BASE, 'lib')
    incl = os.path.join(site.USER_BASE, 'include')
    os.mkdir(lib)
    os.mkdir(incl)
    cmd.ensure_finalized()
    self.assertIn(lib, cmd.library_dirs)
    self.assertIn(lib, cmd.rpath)
    self.assertIn(incl, cmd.include_dirs)