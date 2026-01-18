from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_splititer(self):
    self.assertEqual(regex.split(',', 'a,b,,c,'), ['a', 'b', '', 'c', ''])
    self.assertEqual([m for m in regex.splititer(',', 'a,b,,c,')], ['a', 'b', '', 'c', ''])