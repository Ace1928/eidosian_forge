from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_qualified_re_split(self):
    self.assertEqual(regex.split(':', ':a:b::c', 2), ['', 'a', 'b::c'])
    self.assertEqual(regex.split(':', 'a:b:c:d', 2), ['a', 'b', 'c:d'])
    self.assertEqual(regex.split('(:)', ':a:b::c', 2), ['', ':', 'a', ':', 'b::c'])
    if sys.version_info >= (3, 7, 0):
        self.assertEqual(regex.split('(:*)', ':a:b::c', 2), ['', ':', '', '', 'a:b::c'])
    else:
        self.assertEqual(regex.split('(:*)', ':a:b::c', 2), ['', ':', 'a', ':', 'b::c'])