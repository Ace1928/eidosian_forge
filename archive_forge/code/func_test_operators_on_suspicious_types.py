import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_operators_on_suspicious_types(self):

    class Spam(numbers.Number):

        def __add__(inner_self, other):
            self.fail('doing attribute lookup might have side effects')
    with self.assertRaises(ValueError):
        simple_eval('a + 1', {'a': Spam()})