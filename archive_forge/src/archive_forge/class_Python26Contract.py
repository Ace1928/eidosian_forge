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
class Python26Contract(TestControlContract):

    def test_fresh_result_is_successful(self):
        result = self.makeResult()
        self.assertTrue(result.wasSuccessful())

    def test_addError_is_failure(self):
        result = self.makeResult()
        result.startTest(self)
        result.addError(self, an_exc_info)
        result.stopTest(self)
        self.assertFalse(result.wasSuccessful())

    def test_addFailure_is_failure(self):
        result = self.makeResult()
        result.startTest(self)
        result.addFailure(self, an_exc_info)
        result.stopTest(self)
        self.assertFalse(result.wasSuccessful())

    def test_addSuccess_is_success(self):
        result = self.makeResult()
        result.startTest(self)
        result.addSuccess(self)
        result.stopTest(self)
        self.assertTrue(result.wasSuccessful())