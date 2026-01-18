import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_nonexistant_names_raise(self):
    with self.assertRaises(EvaluationError):
        simple_eval('a')