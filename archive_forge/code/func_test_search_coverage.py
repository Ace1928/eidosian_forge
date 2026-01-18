from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_search_coverage(self):
    self.assertEqual(regex.search('\\s(b)', ' b')[1], 'b')
    self.assertEqual(regex.search('a\\s', 'a ')[0], 'a ')