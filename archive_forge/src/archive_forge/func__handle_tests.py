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
def _handle_tests(self, test_record):
    case = test_record.to_test_case()
    case.run(self.decorated)