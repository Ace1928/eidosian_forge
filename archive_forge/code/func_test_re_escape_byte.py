from weakref import proxy
import copy
import pickle
import regex
import string
import sys
import unittest
def test_re_escape_byte(self):
    p = b''
    self.assertEqual(regex.escape(p), p)
    for i in range(0, 256):
        b = bytes([i])
        p += b
        self.assertEqual(bool(regex.match(regex.escape(b), b)), True)
        self.assertEqual(regex.match(regex.escape(b), b).span(), (0, 1))
    pat = regex.compile(regex.escape(p))
    self.assertEqual(pat.match(p).span(), (0, 256))