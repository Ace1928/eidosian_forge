import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_write_to_file_subsection(self):
    c = ConfigFile()
    c.set((b'branch', b'blie'), b'foo', b'bar')
    f = BytesIO()
    c.write_to_file(f)
    self.assertEqual(b'[branch "blie"]\n\tfoo = bar\n', f.getvalue())