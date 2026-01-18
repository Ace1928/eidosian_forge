import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_with_namespace(self):
    self.assertEvaled('a[1].a|bc', 'd', {'a': 'adsf'})
    self.assertCannotEval('a[1].a|bc', {})