import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.common.tempfiles as tempfiles
from pyomo.common.dependencies import pyutilib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
@unittest.skipIf(not sys.platform.lower().startswith('win'), 'test only applies to Windows platforms')
def test_open_tempfile_windows(self):
    self.TM.push()
    fname = self.TM.create_tempfile()
    f = open(fname)
    try:
        _orig = tempfiles.deletion_errors_are_fatal
        tempfiles.deletion_errors_are_fatal = True
        with self.assertRaisesRegex(WindowsError, '.*process cannot access the file'):
            self.TM.pop()
    finally:
        tempfiles.deletion_errors_are_fatal = _orig
        f.close()
        os.remove(fname)
    self.TM.push()
    fname = self.TM.create_tempfile()
    f = open(fname)
    try:
        _orig = tempfiles.deletion_errors_are_fatal
        tempfiles.deletion_errors_are_fatal = False
        with LoggingIntercept(None, 'pyomo.common') as LOG:
            self.TM.pop()
        self.assertIn('Unable to delete temporary file', LOG.getvalue())
    finally:
        tempfiles.deletion_errors_are_fatal = _orig
        f.close()
        os.remove(fname)