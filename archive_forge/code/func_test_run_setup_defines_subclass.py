import io
import distutils.core
import os
import shutil
import sys
from test.support import captured_stdout
from test.support import os_helper
import unittest
from distutils.tests import support
from distutils import log
from distutils.core import setup
import os
from distutils.core import setup
from distutils.core import setup
from distutils.core import setup
from distutils.command.install import install as _install
def test_run_setup_defines_subclass(self):
    dist = distutils.core.run_setup(self.write_setup(setup_defines_subclass))
    install = dist.get_command_obj('install')
    self.assertIn('cmd', install.sub_commands)