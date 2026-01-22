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
def _expanduser(path):
    return self.tmpdir