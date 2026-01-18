import contextlib
import os
import shutil
import subprocess
import sys
import textwrap
import unittest
from distutils import sysconfig
from distutils.ccompiler import get_default_compiler
from distutils.tests import support
from test.support import swap_item, requires_subprocess, is_wasi
from test.support.os_helper import TESTFN
from test.support.warnings_helper import check_warnings
def test_sysconfig_module(self):
    import sysconfig as global_sysconfig
    self.assertEqual(global_sysconfig.get_config_var('CFLAGS'), sysconfig.get_config_var('CFLAGS'))
    self.assertEqual(global_sysconfig.get_config_var('LDFLAGS'), sysconfig.get_config_var('LDFLAGS'))