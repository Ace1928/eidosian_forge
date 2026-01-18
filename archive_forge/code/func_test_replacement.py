from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_replacement(self):
    self.assertEqual(regex.sub('test\\?', 'result\\?\\.\x07\n', 'test?'), 'result\\?\\.\x07\n')
    self.assertEqual(regex.sub('(.)', '\\1\\1', 'x'), 'xx')
    self.assertEqual(regex.sub('(.)', regex.escape('\\1\\1'), 'x'), '\\1\\1')
    self.assertEqual(regex.sub('(.)', '\\\\1\\\\1', 'x'), '\\1\\1')
    self.assertEqual(regex.sub('(.)', lambda m: '\\1\\1', 'x'), '\\1\\1')