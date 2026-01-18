import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_write_file_text(self):
    text = u'abcüöä'
    self._test_write_file(text, text.encode('utf8'))