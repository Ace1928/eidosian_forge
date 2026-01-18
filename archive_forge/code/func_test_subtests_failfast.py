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
def test_subtests_failfast(self):
    events = []

    class Foo(unittest.TestCase):

        def test_a(self):
            with self.subTest():
                events.append('a1')
            events.append('a2')

        def test_b(self):
            with self.subTest():
                events.append('b1')
            with self.subTest():
                self.fail('failure')
            events.append('b2')

        def test_c(self):
            events.append('c')
    result = unittest.TestResult()
    result.failfast = True
    suite = unittest.TestLoader().loadTestsFromTestCase(Foo)
    suite.run(result)
    expected = ['a1', 'a2', 'b1']
    self.assertEqual(events, expected)