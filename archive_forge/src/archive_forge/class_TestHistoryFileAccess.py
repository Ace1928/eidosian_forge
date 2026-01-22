import tempfile
import unittest
from pathlib import Path
from bpython.config import getpreferredencoding
from bpython.history import History
class TestHistoryFileAccess(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.filename = Path(self.tempdir.name) / 'history_temp_file'
        self.encoding = getpreferredencoding()
        with open(self.filename, 'w', encoding=self.encoding, errors='ignore') as f:
            f.write(b'#1\n#2\n'.decode())

    def test_load(self):
        history = History()
        history.load(self.filename, self.encoding)
        self.assertEqual(history.entries, ['#1', '#2'])

    def test_append_reload_and_write(self):
        history = History()
        history.append_reload_and_write('#3', self.filename, self.encoding)
        self.assertEqual(history.entries, ['#1', '#2', '#3'])
        history.append_reload_and_write('#4', self.filename, self.encoding)
        self.assertEqual(history.entries, ['#1', '#2', '#3', '#4'])

    def test_save(self):
        history = History()
        for line in ('#1', '#2', '#3', '#4'):
            history.append_to(history.entries, line)
        history.save(self.filename, self.encoding, lines=2)
        history = History()
        history.load(self.filename, self.encoding)
        self.assertEqual(history.entries, ['#3', '#4'])

    def tearDown(self):
        self.tempdir = None