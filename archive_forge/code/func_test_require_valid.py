from __future__ import unicode_literals
import contextlib
import difflib
import io
import os
import shutil
import subprocess
import sys
import unittest
import tempfile
def test_require_valid(self):
    """
    Verify that the --require-valid-layout flag works as intended
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    testfilepath = os.path.join(thisdir, 'testdata', 'test_invalid.cmake')
    with tempfile.NamedTemporaryFile(suffix='.txt', prefix='CMakeLists', dir=self.tempdir) as outfile:
        statuscode = subprocess.call([sys.executable, '-Bm', 'cmakelang.format', testfilepath], stdout=outfile, stderr=outfile, env=self.env)
    self.assertEqual(0, statuscode)
    with tempfile.NamedTemporaryFile(suffix='.txt', prefix='CMakeLists', dir=self.tempdir) as outfile:
        statuscode = subprocess.call([sys.executable, '-Bm', 'cmakelang.format', testfilepath, '--require-valid-layout'], stdout=outfile, stderr=outfile, env=self.env)
    self.assertEqual(1, statuscode)