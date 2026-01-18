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
def test_pushpop2_dir(self):
    """Test pushpop logic with directories"""
    TempfileManager.push()
    os.mkdir(tempdir + 'pushpop2')
    TempfileManager.add_tempfile(tempdir + 'pushpop2')
    TempfileManager.push()
    os.mkdir(tempdir + 'pushpop2a')
    TempfileManager.add_tempfile(tempdir + 'pushpop2a')
    TempfileManager.pop()
    if not os.path.exists(tempdir + 'pushpop2'):
        self.fail('pop() clean out all files')
    if os.path.exists(tempdir + 'pushpop2a'):
        self.fail('pop() failed to clean out files')
    TempfileManager.pop()
    if os.path.exists(tempdir + 'pushpop2'):
        self.fail('pop() failed to clean out files')