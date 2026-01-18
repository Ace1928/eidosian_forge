import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_one_none_completer_returns_none(self):
    a = self.completer(None)
    cumulative = autocomplete.CumulativeCompleter([a])
    self.assertEqual(cumulative.matches(3, 'abc'), None)