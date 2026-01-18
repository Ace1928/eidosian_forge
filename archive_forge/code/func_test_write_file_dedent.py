import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_write_file_dedent(self):
    text = u'\n        A horse is a horse,\n        of course, of course,\n        And no one can talk to a horse\n        of course\n        '
    self._test_write_file(text, textwrap.dedent(text).encode('utf8'), dedent=True)