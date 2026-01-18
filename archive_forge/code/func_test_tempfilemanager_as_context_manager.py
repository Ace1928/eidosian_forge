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
def test_tempfilemanager_as_context_manager(self):
    with LoggingIntercept() as LOG:
        with self.TM:
            fname = self.TM.create_tempfile()
            self.assertTrue(os.path.exists(fname))
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(LOG.getvalue(), '')
        with self.TM:
            self.TM.push()
            fname = self.TM.create_tempfile()
            self.assertTrue(os.path.exists(fname))
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(LOG.getvalue().strip(), 'TempfileManager: tempfile context was pushed onto the TempfileManager stack within a context manager (i.e., `with TempfileManager:`) but was not popped before the context manager exited.  Popping the context to preserve the stack integrity.')