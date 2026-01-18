import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_returns_bytes(self):
    u = 'Hello'
    encoded_string = encode_string(u)
    self.assertIsInstance(encoded_string, bytes)