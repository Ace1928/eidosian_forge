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
def test_parse_makefile_literal_dollar(self):
    self.makefile = TESTFN
    fd = open(self.makefile, 'w')
    try:
        fd.write("CONFIG_ARGS=  '--arg1=optarg1' 'ENV=\\$$LIB'\n")
        fd.write('VAR=$OTHER\nOTHER=foo')
    finally:
        fd.close()
    d = sysconfig.parse_makefile(self.makefile)
    self.assertEqual(d, {'CONFIG_ARGS': "'--arg1=optarg1' 'ENV=\\$LIB'", 'OTHER': 'foo'})