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
@requires_subprocess()
def test_record_extensions(self):
    cmd = test_support.missing_compiler_executable()
    if cmd is not None:
        self.skipTest('The %r command is not found' % cmd)
    install_dir = self.mkdtemp()
    project_dir, dist = self.create_dist(ext_modules=[Extension('xx', ['xxmodule.c'])])
    os.chdir(project_dir)
    support.copy_xxmodule_c(project_dir)
    buildextcmd = build_ext(dist)
    support.fixup_build_ext(buildextcmd)
    buildextcmd.ensure_finalized()
    cmd = install(dist)
    dist.command_obj['install'] = cmd
    dist.command_obj['build_ext'] = buildextcmd
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
    expected = [_make_ext_name('xx'), 'UNKNOWN-0.0.0-py%s.%s.egg-info' % sys.version_info[:2]]
    self.assertEqual(found, expected)