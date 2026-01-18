import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_smp(self):
    utf_8 = b'a\xf0\x93\x82\xa7b'
    u = 'a𓂧b'
    b = encode_string(u, 'utf-8', 'strict', AssertionError)
    self.assertEqual(b, utf_8)