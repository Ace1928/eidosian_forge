import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_path_with_null_bytes(self):
    b = b'a\x00b\x00c'
    encoded_file_path = encode_file_path(b)
    self.assertIsNone(encoded_file_path)