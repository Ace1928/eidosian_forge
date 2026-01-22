import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _gather_test(self, test_record):
    if test_record.status == 'exists':
        return
    self.testsRun += 1
    case = test_record.to_test_case()
    self._handle_status[test_record.status](case)