import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestGlobalCompletion(unittest.TestCase):

    def setUp(self):
        self.com = autocomplete.GlobalCompletion()

    def test_function(self):

        def function():
            pass
        self.assertEqual(self.com.matches(8, 'function', locals_={'function': function}), {'function('})

    def test_completions_are_unicode(self):
        for m in self.com.matches(1, 'a', locals_={'abc': 10}):
            self.assertIsInstance(m, str)

    def test_mock_kwlist(self):
        with mock.patch.object(keyword, 'kwlist', new=['abcd']):
            self.assertEqual(self.com.matches(3, 'abc', locals_={}), None)

    def test_mock_kwlist_non_ascii(self):
        with mock.patch.object(keyword, 'kwlist', new=['abcß']):
            self.assertEqual(self.com.matches(3, 'abc', locals_={}), None)