from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_python_module(self):
    """Test completing a module run with python -m."""
    prog = os.path.join(TEST_DIR, 'prog')
    with TempDir(prefix='test_dir_py', dir='.'):
        os.mkdir('package')
        open('package/__init__.py', 'w').close()
        shutil.copy(prog, 'package/prog.py')
        self.sh.run_command('cd ' + os.getcwd())
        self.assertEqual(self.sh.run_command('python -m package.prog basic f\t'), 'foo\r\n')