import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_reset(self):
    self.history.enter('#lastnumber!')
    self.history.reset()
    self.assertEqual(self.history.back(), '#999')
    self.assertEqual(self.history.forward(), '')