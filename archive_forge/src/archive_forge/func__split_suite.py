from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
@staticmethod
def _split_suite(suite):
    serial = unittest.TestSuite()
    parallel = unittest.TestSuite()
    for test in suite:
        if test.countTestCases() == 0:
            continue
        if isinstance(test, unittest.TestSuite):
            test_class = test._tests[0].__class__
        elif isinstance(test, unittest.TestCase):
            test_class = test
        else:
            raise TypeError("can't recognize type %r" % test)
        if getattr(test_class, '_serialrun', False):
            serial.addTest(test)
        else:
            parallel.addTest(test)
    return (serial, parallel)