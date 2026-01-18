import pathlib
import unittest
from pygame import encode_string, encode_file_path
def test_obj_bytes(self):
    b = b'encyclop\xe6dia'
    encoded_string = encode_string(b, 'ascii', 'strict')
    self.assertIs(encoded_string, b)