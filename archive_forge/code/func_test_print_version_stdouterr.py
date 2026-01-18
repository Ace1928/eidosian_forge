import sys
import unittest
from Cython.Utils import (
def test_print_version_stdouterr(self):
    orig_stderr = sys.stderr
    orig_stdout = sys.stdout
    stdout = sys.stdout = sys.stderr = StringIO()
    try:
        print_version()
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    stdout = stdout.getvalue()
    from .. import __version__ as version
    self.assertIn(version, stdout)
    self.assertEqual(stdout.count(version), 1)