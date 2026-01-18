from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_repeated_repeats(self):
    self.assertEqual(regex.search('(?:a+)+', 'aaa').span(), (0, 3))
    self.assertEqual(regex.search('(?:(?:ab)+c)+', 'abcabc').span(), (0, 6))
    self.assertEqual(regex.search('(?:a+){2,}', 'aaa').span(), (0, 3))