import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def _patch_standard_output(self):
    """Replace the stdout and stderr streams with string-based streams
        in order to capture the tests' output.
        """
    if not self.output_patched:
        self.old_stdout, self.old_stderr = (sys.stdout, sys.stderr)
        self.output_patched = True
    sys.stdout, sys.stderr = self.stdout, self.stderr = (StringIO(), StringIO())