import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_write_preserve_multivar(self):
    cf = self.from_file(b'[core]\nfoo = bar\nfoo = blah\n')
    f = BytesIO()
    cf.write_to_file(f)
    self.assertEqual(b'[core]\n\tfoo = bar\n\tfoo = blah\n', f.getvalue())