import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_write_newer_file_fresh(self):
    file_path = self._test_path('abcfile.txt')
    assert not os.path.exists(file_path)
    write_newer_file(file_path, file_path, 'xyz')
    assert os.path.isfile(file_path)