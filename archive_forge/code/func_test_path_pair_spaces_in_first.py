import io
import time
import unittest
from fastimport import (
from :2
def test_path_pair_spaces_in_first(self):
    p = parser.ImportParser('')
    self.assertEqual([b'foo bar', b'baz'], p._path_pair(b'"foo bar" baz'))