import os
import sys
import unittest
import site
from test.support import captured_stdout, requires_subprocess
from distutils import sysconfig
from distutils.command.install import install, HAS_USER_SITE
from distutils.command import install as install_module
from distutils.command.build_ext import build_ext
from distutils.command.install import INSTALL_SCHEMES
from distutils.core import Distribution
from distutils.errors import DistutilsOptionError
from distutils.extension import Extension
from distutils.tests import support
from test import support as test_support
def test_record(self):
    install_dir = self.mkdtemp()
    project_dir, dist = self.create_dist(py_modules=['hello'], scripts=['sayhi'])
    os.chdir(project_dir)
    self.write_file('hello.py', "def main(): print('o hai')")
    self.write_file('sayhi', 'from hello import main; main()')
    cmd = install(dist)
    dist.command_obj['install'] = cmd
    cmd.root = install_dir
    cmd.record = os.path.join(project_dir, 'filelist')
    cmd.ensure_finalized()
    cmd.run()
    f = open(cmd.record)
    try:
        content = f.read()
    finally:
        f.close()
    found = [os.path.basename(line) for line in content.splitlines()]
    expected = ['hello.py', 'hello.%s.pyc' % sys.implementation.cache_tag, 'sayhi', 'UNKNOWN-0.0.0-py%s.%s.egg-info' % sys.version_info[:2]]
    self.assertEqual(found, expected)