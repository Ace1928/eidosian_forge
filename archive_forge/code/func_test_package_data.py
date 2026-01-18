import os
import sys
import unittest
from distutils.command.build_py import build_py
from distutils.core import Distribution
from distutils.errors import DistutilsFileError
from distutils.tests import support
from test.support import requires_subprocess
def test_package_data(self):
    sources = self.mkdtemp()
    f = open(os.path.join(sources, '__init__.py'), 'w')
    try:
        f.write('# Pretend this is a package.')
    finally:
        f.close()
    f = open(os.path.join(sources, 'README.txt'), 'w')
    try:
        f.write('Info about this package')
    finally:
        f.close()
    destination = self.mkdtemp()
    dist = Distribution({'packages': ['pkg'], 'package_dir': {'pkg': sources}})
    dist.script_name = os.path.join(sources, 'setup.py')
    dist.command_obj['build'] = support.DummyCommand(force=0, build_lib=destination)
    dist.packages = ['pkg']
    dist.package_data = {'pkg': ['README.txt']}
    dist.package_dir = {'pkg': sources}
    cmd = build_py(dist)
    cmd.compile = 1
    cmd.ensure_finalized()
    self.assertEqual(cmd.package_data, dist.package_data)
    cmd.run()
    self.assertEqual(len(cmd.get_outputs()), 3)
    pkgdest = os.path.join(destination, 'pkg')
    files = os.listdir(pkgdest)
    pycache_dir = os.path.join(pkgdest, '__pycache__')
    self.assertIn('__init__.py', files)
    self.assertIn('README.txt', files)
    if sys.dont_write_bytecode:
        self.assertFalse(os.path.exists(pycache_dir))
    else:
        pyc_files = os.listdir(pycache_dir)
        self.assertIn('__init__.%s.pyc' % sys.implementation.cache_tag, pyc_files)