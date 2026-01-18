import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_write_file_bytes(self):
    self._test_write_file(b'ab\x00c', b'ab\x00c')