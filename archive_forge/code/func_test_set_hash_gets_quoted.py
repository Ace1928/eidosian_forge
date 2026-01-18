import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_set_hash_gets_quoted(self):
    c = ConfigFile()
    c.set(b'xandikos', b'color', b'#665544')
    f = BytesIO()
    c.write_to_file(f)
    self.assertEqual(b'[xandikos]\n\tcolor = "#665544"\n', f.getvalue())