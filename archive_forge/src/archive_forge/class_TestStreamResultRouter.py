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
class TestStreamResultRouter(TestCase):

    def test_start_stop_test_run_no_fallback(self):
        result = StreamResultRouter()
        result.startTestRun()
        result.stopTestRun()

    def test_no_fallback_errors(self):
        self.assertRaises(Exception, StreamResultRouter().status, test_id='f')

    def test_fallback_calls(self):
        fallback = LoggingStreamResult()
        result = StreamResultRouter(fallback)
        result.startTestRun()
        result.status(test_id='foo')
        result.stopTestRun()
        self.assertEqual([('startTestRun',), ('status', 'foo', None, None, True, None, None, False, None, None, None), ('stopTestRun',)], fallback._events)

    def test_fallback_no_do_start_stop_run(self):
        fallback = LoggingStreamResult()
        result = StreamResultRouter(fallback, do_start_stop_run=False)
        result.startTestRun()
        result.status(test_id='foo')
        result.stopTestRun()
        self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, None, None)], fallback._events)

    def test_add_rule_bad_policy(self):
        router = StreamResultRouter()
        target = LoggingStreamResult()
        self.assertRaises(ValueError, router.add_rule, target, 'route_code_prefixa', route_prefix='0')

    def test_add_rule_extra_policy_arg(self):
        router = StreamResultRouter()
        target = LoggingStreamResult()
        self.assertRaises(TypeError, router.add_rule, target, 'route_code_prefix', route_prefix='0', foo=1)

    def test_add_rule_missing_prefix(self):
        router = StreamResultRouter()
        target = LoggingStreamResult()
        self.assertRaises(TypeError, router.add_rule, target, 'route_code_prefix')

    def test_add_rule_slash_in_prefix(self):
        router = StreamResultRouter()
        target = LoggingStreamResult()
        self.assertRaises(TypeError, router.add_rule, target, 'route_code_prefix', route_prefix='0/')

    def test_add_rule_route_code_consume_False(self):
        fallback = LoggingStreamResult()
        target = LoggingStreamResult()
        router = StreamResultRouter(fallback)
        router.add_rule(target, 'route_code_prefix', route_prefix='0')
        router.status(test_id='foo', route_code='0')
        router.status(test_id='foo', route_code='0/1')
        router.status(test_id='foo')
        self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, '0', None), ('status', 'foo', None, None, True, None, None, False, None, '0/1', None)], target._events)
        self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, None, None)], fallback._events)

    def test_add_rule_route_code_consume_True(self):
        fallback = LoggingStreamResult()
        target = LoggingStreamResult()
        router = StreamResultRouter(fallback)
        router.add_rule(target, 'route_code_prefix', route_prefix='0', consume_route=True)
        router.status(test_id='foo', route_code='0')
        router.status(test_id='foo', route_code='0/1')
        router.status(test_id='foo', route_code='1')
        self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, None, None), ('status', 'foo', None, None, True, None, None, False, None, '1', None)], target._events)
        self.assertEqual([('status', 'foo', None, None, True, None, None, False, None, '1', None)], fallback._events)

    def test_add_rule_test_id(self):
        nontest = LoggingStreamResult()
        test = LoggingStreamResult()
        router = StreamResultRouter(test)
        router.add_rule(nontest, 'test_id', test_id=None)
        router.status(test_id='foo', file_name='bar', file_bytes=b'')
        router.status(file_name='bar', file_bytes=b'')
        self.assertEqual([('status', 'foo', None, None, True, 'bar', b'', False, None, None, None)], test._events)
        self.assertEqual([('status', None, None, None, True, 'bar', b'', False, None, None, None)], nontest._events)

    def test_add_rule_do_start_stop_run(self):
        nontest = LoggingStreamResult()
        router = StreamResultRouter()
        router.add_rule(nontest, 'test_id', test_id=None, do_start_stop_run=True)
        router.startTestRun()
        router.stopTestRun()
        self.assertEqual([('startTestRun',), ('stopTestRun',)], nontest._events)

    def test_add_rule_do_start_stop_run_after_startTestRun(self):
        nontest = LoggingStreamResult()
        router = StreamResultRouter()
        router.startTestRun()
        router.add_rule(nontest, 'test_id', test_id=None, do_start_stop_run=True)
        router.stopTestRun()
        self.assertEqual([('startTestRun',), ('stopTestRun',)], nontest._events)