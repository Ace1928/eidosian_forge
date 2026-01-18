import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_operators_on_numbers(self):
    self.assertEqual(simple_eval('-2'), -2)
    self.assertEqual(simple_eval('1 + 1'), 2)
    self.assertEqual(simple_eval('a - 2', {'a': 1}), -1)
    with self.assertRaises(ValueError):
        simple_eval('2 * 3')
    with self.assertRaises(ValueError):
        simple_eval('2 ** 3')