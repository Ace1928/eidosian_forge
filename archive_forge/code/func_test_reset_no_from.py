from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_reset_no_from(self):
    c = commands.ResetCommand(b'refs/remotes/origin/master', None)
    self.assertEqual(b'reset refs/remotes/origin/master', bytes(c))