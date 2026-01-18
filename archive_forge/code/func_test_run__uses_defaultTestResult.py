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
def test_run__uses_defaultTestResult(self):
    events = []
    defaultResult = LoggingResult(events)

    class Foo(unittest.TestCase):

        def test(self):
            events.append('test')

        def defaultTestResult(self):
            return defaultResult
    result = Foo('test').run()
    self.assertIs(result, defaultResult)
    expected = ['startTestRun', 'startTest', 'test', 'addSuccess', 'stopTest', 'stopTestRun']
    self.assertEqual(events, expected)