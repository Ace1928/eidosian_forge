import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_one_empty_completer_returns_empty(self):
    a = self.completer([])
    cumulative = autocomplete.CumulativeCompleter([a])
    self.assertEqual(cumulative.matches(3, 'abc'), set())