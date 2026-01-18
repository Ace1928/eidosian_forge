from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_scoped_and_inline_flags(self):
    self.assertEqual(regex.search('(?i)Ab', 'ab').span(), (0, 2))
    self.assertEqual(regex.search('(?i:A)b', 'ab').span(), (0, 2))
    self.assertEqual(regex.search('A(?i)b', 'ab'), None)
    self.assertEqual(regex.search('(?V0)Ab', 'ab'), None)
    self.assertEqual(regex.search('(?V1)Ab', 'ab'), None)
    self.assertEqual(regex.search('(?-i)Ab', 'ab', flags=regex.I), None)
    self.assertEqual(regex.search('(?-i:A)b', 'ab', flags=regex.I), None)
    self.assertEqual(regex.search('A(?-i)b', 'ab', flags=regex.I).span(), (0, 2))