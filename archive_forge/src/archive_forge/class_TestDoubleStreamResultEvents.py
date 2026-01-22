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
class TestDoubleStreamResultEvents(TestCase):

    def test_startTestRun(self):
        result = LoggingStreamResult()
        result.startTestRun()
        self.assertEqual([('startTestRun',)], result._events)

    def test_stopTestRun(self):
        result = LoggingStreamResult()
        result.startTestRun()
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('stopTestRun',)], result._events)

    def test_file(self):
        result = LoggingStreamResult()
        result.startTestRun()
        now = datetime.datetime.now(utc)
        result.status(file_name='foo', file_bytes='bar', eof=True, mime_type='text/json', test_id='id', route_code='abc', timestamp=now)
        self.assertEqual([('startTestRun',), ('status', 'id', None, None, True, 'foo', 'bar', True, 'text/json', 'abc', now)], result._events)

    def test_status(self):
        result = LoggingStreamResult()
        result.startTestRun()
        now = datetime.datetime.now(utc)
        result.status('foo', 'success', test_tags={'tag'}, runnable=False, route_code='abc', timestamp=now)
        self.assertEqual([('startTestRun',), ('status', 'foo', 'success', {'tag'}, False, None, None, False, None, 'abc', now)], result._events)