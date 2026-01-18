import os
import shutil
import pytest
from tempfile import mkstemp, mkdtemp
from subprocess import Popen, PIPE
import importlib.metadata
from distutils.errors import DistutilsError
from numpy.testing import assert_, assert_equal, assert_raises
from numpy.distutils import ccompiler, customized_ccompiler
from numpy.distutils.system_info import system_info, ConfigParser, mkl_info
from numpy.distutils.system_info import AliasedOptionError
from numpy.distutils.system_info import default_lib_dirs, default_include_dirs
from numpy.distutils import _shell_utils
def test_temp1(self):
    tsi = self.c_temp1
    assert_equal(tsi.get_lib_dirs(), [self._dir1])
    assert_equal(tsi.get_libraries(), [self._lib1])
    assert_equal(tsi.get_runtime_lib_dirs(), [self._dir1])