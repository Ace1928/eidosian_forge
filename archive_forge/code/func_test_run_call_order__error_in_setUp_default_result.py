import contextlib
import difflib
import pprint
import pickle
import re
import sys
import logging
import warnings
import weakref
import inspect
import types
from copy import deepcopy
from test import support
import unittest
from unittest.test.support import (
from test.support import captured_stderr, gc_collect
def test_run_call_order__error_in_setUp_default_result(self):
    events = []

    class Foo(Test.LoggingTestCase):

        def defaultTestResult(self):
            return LoggingResult(self.events)

        def setUp(self):
            super(Foo, self).setUp()
            raise RuntimeError('raised by Foo.setUp')
    Foo(events).run()
    expected = ['startTestRun', 'startTest', 'setUp', 'addError', 'stopTest', 'stopTestRun']
    self.assertEqual(events, expected)