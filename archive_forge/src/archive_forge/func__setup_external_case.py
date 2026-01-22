import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
def _setup_external_case(self, testline, coding='ascii', modulelevel='', suffix=''):
    """Create a test case in a separate module"""
    _, prefix, self.modname = self.id().rsplit('.', 2)
    self.dir = tempfile.mkdtemp(prefix=prefix, suffix=suffix)
    self.addCleanup(shutil.rmtree, self.dir)
    self._write_module(self.modname, coding, '# coding: %s\nimport testtools\n%s\nclass Test(testtools.TestCase):\n    def runTest(self):\n        %s\n' % (coding, modulelevel, testline))