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
class TestStreamToDict(TestCase):

    def test_hung_test(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status('foo', 'inprogress')
        self.assertEqual([], tests)
        result.stopTestRun()
        self.assertEqual([{'id': 'foo', 'tags': set(), 'details': {}, 'status': 'inprogress', 'timestamps': [None, None]}], tests)

    def test_all_terminal_states_reported(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status('success', 'success')
        result.status('skip', 'skip')
        result.status('exists', 'exists')
        result.status('fail', 'fail')
        result.status('xfail', 'xfail')
        result.status('uxsuccess', 'uxsuccess')
        self.assertThat(tests, HasLength(6))
        self.assertEqual(['success', 'skip', 'exists', 'fail', 'xfail', 'uxsuccess'], [test['id'] for test in tests])
        result.stopTestRun()
        self.assertThat(tests, HasLength(6))

    def test_files_reported(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status(file_name='some log.txt', file_bytes=_b('1234 log message'), eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
        result.status(file_name='another file', file_bytes=_b('Traceback...'), test_id='foo.bar')
        result.stopTestRun()
        self.assertThat(tests, HasLength(1))
        test = tests[0]
        self.assertEqual('foo.bar', test['id'])
        self.assertEqual('unknown', test['status'])
        details = test['details']
        self.assertEqual('1234 log message', details['some log.txt'].as_text())
        self.assertEqual(_b('Traceback...'), _b('').join(details['another file'].iter_bytes()))
        self.assertEqual('application/octet-stream', repr(details['another file'].content_type))

    def test_bad_mime(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status(file_name='file', file_bytes=b'a', mime_type='text/plain; charset=utf8, language=python', test_id='id')
        result.stopTestRun()
        self.assertThat(tests, HasLength(1))
        test = tests[0]
        self.assertEqual('id', test['id'])
        details = test['details']
        self.assertEqual('a', details['file'].as_text())
        self.assertEqual('text/plain; charset="utf8"', repr(details['file'].content_type))

    def test_timestamps(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status(test_id='foo', test_status='inprogress', timestamp='A')
        result.status(test_id='foo', test_status='success', timestamp='B')
        result.status(test_id='bar', test_status='inprogress', timestamp='C')
        result.stopTestRun()
        self.assertThat(tests, HasLength(2))
        self.assertEqual(['A', 'B'], tests[0]['timestamps'])
        self.assertEqual(['C', None], tests[1]['timestamps'])

    def test_files_skipped(self):
        tests = []
        result = StreamToDict(tests.append)
        result.startTestRun()
        result.status(file_name='some log.txt', file_bytes='', eof=True, mime_type='text/plain; charset=utf8', test_id='foo.bar')
        result.stopTestRun()
        self.assertThat(tests, HasLength(1))
        details = tests[0]['details']
        self.assertNotIn('some log.txt', details)