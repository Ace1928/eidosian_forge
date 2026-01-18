import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_two_completers_with_matches_returns_first_matches(self):
    a = completer(['a'])
    b = completer(['b'])
    self.assertEqual(autocomplete.get_completer([a, b], 0, ''), (['a'], a))