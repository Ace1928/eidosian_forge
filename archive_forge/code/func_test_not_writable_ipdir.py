import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
@dec.skip_win32
@with_environment
def test_not_writable_ipdir(self):
    tmpdir = tempfile.mkdtemp()
    os.name = 'posix'
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)
    env['HOME'] = tmpdir
    ipdir = os.path.join(tmpdir, '.ipython')
    os.mkdir(ipdir, 365)
    try:
        open(os.path.join(ipdir, '_foo_'), 'w', encoding='utf-8').close()
    except IOError:
        pass
    else:
        pytest.skip("I can't create directories that I can't write to")
    with self.assertWarnsRegex(UserWarning, 'is not a writable location'):
        ipdir = paths.get_ipython_dir()
    env.pop('IPYTHON_DIR', None)