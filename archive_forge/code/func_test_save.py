import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
def test_save(self):
    history = History()
    for line in ('#1', '#2', '#3', '#4'):
        history.append_to(history.entries, line)
    history.save(self.filename, self.encoding, lines=2)
    history = History()
    history.load(self.filename, self.encoding)
    self.assertEqual(history.entries, ['#3', '#4'])