from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_escape(self):
    p = ''
    self.assertEqual(regex.escape(p), p)
    for i in range(0, 256):
        p += chr(i)
        self.assertEqual(bool(regex.match(regex.escape(chr(i)), chr(i))), True)
        self.assertEqual(regex.match(regex.escape(chr(i)), chr(i)).span(), (0, 1))
    pat = regex.compile(regex.escape(p))
    self.assertEqual(pat.match(p).span(), (0, 256))