import io
import time
import unittest
from fastimport import (
from :2
def test_path_pair_simple(self):
    p = parser.ImportParser(b'')
    self.assertEqual([b'foo', b'bar'], p._path_pair(b'foo bar'))