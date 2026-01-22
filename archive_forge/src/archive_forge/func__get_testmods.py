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
def _get_testmods(self):
    return [os.path.join(self.testdir, x) for x in os.listdir(self.testdir) if x.startswith('test_') and x.endswith('.py') and (x not in self.skip_files)]