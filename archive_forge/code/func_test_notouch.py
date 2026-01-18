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
def test_notouch(self):
    """
    Verify that, if formatting is unchanged, an --in-place file is not modified
    """
    thisdir = os.path.realpath(os.path.dirname(__file__))
    expectfile_path = os.path.join(thisdir, 'testdata', 'test_out.cmake')
    outfile_path = os.path.join(self.tempdir, 'test_out.cmake')
    shutil.copy2(expectfile_path, outfile_path)
    mtime_before = os.path.getmtime(outfile_path)
    subprocess.check_call([sys.executable, '-Bm', 'cmakelang.format', '-i', outfile_path], cwd=self.tempdir, env=self.env)
    mtime_after = os.path.getmtime(outfile_path)
    self.assertEqual(mtime_before, mtime_after)