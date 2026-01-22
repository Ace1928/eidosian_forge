import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestSafeEval(unittest.TestCase):

    def test_catches_syntax_error(self):
        with self.assertRaises(autocomplete.EvaluationError):
            autocomplete.safe_eval('1re', {})