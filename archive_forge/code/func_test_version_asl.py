import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
from pyomo.core import (
def test_version_asl(self):
    self.assertTrue(self.asl.version() is not None)
    self.assertTrue(type(self.asl.version()) is tuple)
    self.assertEqual(len(self.asl.version()), 4)