import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def _test_write_file(self, content, expected, **kwargs):
    file_path = self._test_path('abcfile')
    write_file(file_path, content, **kwargs)
    assert os.path.isfile(file_path)
    with open(file_path, 'rb') as f:
        found = f.read()
    assert found == expected, (repr(expected), repr(found))