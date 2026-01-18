from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_subn(self):
    self.assertEqual(regex.subn('(?i)b+', 'x', 'bbbb BBBB'), ('x x', 2))
    self.assertEqual(regex.subn('b+', 'x', 'bbbb BBBB'), ('x BBBB', 1))
    self.assertEqual(regex.subn('b+', 'x', 'xyz'), ('xyz', 0))
    self.assertEqual(regex.subn('b*', 'x', 'xyz'), ('xxxyxzx', 4))
    self.assertEqual(regex.subn('b*', 'x', 'xyz', 2), ('xxxyz', 2))