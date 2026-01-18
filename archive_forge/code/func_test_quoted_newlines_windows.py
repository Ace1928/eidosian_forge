import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_quoted_newlines_windows(self):
    cf = self.from_file(b'[alias]\r\nc = \'!f() { \\\r\n printf \'[git commit -m \\"%s\\"]\\n\' \\"$*\\" && \\\r\n git commit -m \\"$*\\"; \\\r\n }; f\'\r\n')
    self.assertEqual(list(cf.sections()), [(b'alias',)])
    self.assertEqual(b'\'!f() { printf \'[git commit -m "%s"]\n\' "$*" && git commit -m "$*"', cf.get((b'alias',), b'c'))