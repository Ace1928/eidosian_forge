import sys
import unittest
from test.support.os_helper import EnvironmentVarGuard
from distutils import sysconfig
from distutils.unixccompiler import UnixCCompiler
Tests for distutils.unixccompiler.