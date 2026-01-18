import os.path
import unittest
import tempfile
import textwrap
import shutil
from ..TestUtils import write_file, write_newer_file, _parse_pattern
def test_parse_pattern(self):
    self.assertEqual(_parse_pattern('pattern'), (None, None, 'pattern'))
    self.assertEqual(_parse_pattern('/start/:pattern'), ('start', None, 'pattern'))
    self.assertEqual(_parse_pattern(':/end/  pattern'), (None, 'end', 'pattern'))
    self.assertEqual(_parse_pattern('/start/:/end/  pattern'), ('start', 'end', 'pattern'))
    self.assertEqual(_parse_pattern('/start/:/end/pattern'), ('start', 'end', 'pattern'))