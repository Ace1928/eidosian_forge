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
class TestExtendedToStreamDecorator(TestCase):

    def test_explicit_time(self):
        log = LoggingStreamResult()
        result = ExtendedToStreamDecorator(log)
        result.startTestRun()
        now = datetime.datetime.now(utc)
        result.time(now)
        result.startTest(self)
        result.addSuccess(self)
        result.stopTest(self)
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('status', 'testtools.tests.test_testresult.TestExtendedToStreamDecorator.test_explicit_time', 'inprogress', None, True, None, None, False, None, None, now), ('status', 'testtools.tests.test_testresult.TestExtendedToStreamDecorator.test_explicit_time', 'success', set(), True, None, None, False, None, None, now), ('stopTestRun',)], log._events)

    def test_wasSuccessful_after_stopTestRun(self):
        log = LoggingStreamResult()
        result = ExtendedToStreamDecorator(log)
        result.startTestRun()
        result.status(test_id='foo', test_status='fail')
        result.stopTestRun()
        self.assertEqual(False, result.wasSuccessful())

    def test_empty_detail_status_correct(self):
        log = LoggingStreamResult()
        result = ExtendedToStreamDecorator(log)
        result.startTestRun()
        now = datetime.datetime.now(utc)
        result.time(now)
        result.startTest(self)
        result.addError(self, details={'foo': text_content('')})
        result.stopTest(self)
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('status', 'testtools.tests.test_testresult.TestExtendedToStreamDecorator.test_empty_detail_status_correct', 'inprogress', None, True, None, None, False, None, None, now), ('status', 'testtools.tests.test_testresult.TestExtendedToStreamDecorator.test_empty_detail_status_correct', None, None, True, 'foo', _b(''), True, 'text/plain; charset="utf8"', None, now), ('status', 'testtools.tests.test_testresult.TestExtendedToStreamDecorator.test_empty_detail_status_correct', 'fail', set(), True, None, None, False, None, None, now), ('stopTestRun',)], log._events)