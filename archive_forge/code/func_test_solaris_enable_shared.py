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
def test_solaris_enable_shared(self):
    dist = Distribution({'name': 'xx'})
    cmd = self.build_ext(dist)
    old = sys.platform
    sys.platform = 'sunos'
    from distutils.sysconfig import _config_vars
    old_var = _config_vars.get('Py_ENABLE_SHARED')
    _config_vars['Py_ENABLE_SHARED'] = 1
    try:
        cmd.ensure_finalized()
    finally:
        sys.platform = old
        if old_var is None:
            del _config_vars['Py_ENABLE_SHARED']
        else:
            _config_vars['Py_ENABLE_SHARED'] = old_var
    self.assertGreater(len(cmd.library_dirs), 0)