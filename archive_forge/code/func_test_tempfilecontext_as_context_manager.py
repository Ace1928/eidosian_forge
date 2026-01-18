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
def test_tempfilecontext_as_context_manager(self):
    with LoggingIntercept() as LOG:
        ctx = self.TM.new_context()
        with ctx:
            fname = ctx.create_tempfile()
            self.assertTrue(os.path.exists(fname))
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(LOG.getvalue(), '')