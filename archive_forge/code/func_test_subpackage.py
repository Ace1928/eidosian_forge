from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_subpackage(self):
    os.mkdir('package')
    self._mkfile('package/__init__.py')
    os.mkdir('package/subpackage')
    self._mkfile('package/subpackage/__init__.py')
    self._mkfile('package/subpackage/module.py')
    path = _check_module.find('package.subpackage.module')
    self.assertEqual(path, os.path.abspath('package/subpackage/module.py'))
    self.assertNotIn('package', sys.modules)
    self.assertNotIn('package.subpackage', sys.modules)
    self.assertNotIn('package.subpackage.module', sys.modules)