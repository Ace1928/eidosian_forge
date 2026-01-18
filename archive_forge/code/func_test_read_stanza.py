import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_read_stanza(self):
    """Load stanza from string"""
    lines = b'revision: mbp@sourcefrog.net-123-abc\ntimestamp: 1130653962\ntimezone: 36000\ncommitter: Martin Pool <mbp@test.sourcefrog.net>\n'.splitlines(True)
    s = read_stanza(lines)
    self.assertTrue('revision' in s)
    self.assertEqual(s.get('revision'), 'mbp@sourcefrog.net-123-abc')
    self.assertEqual(list(s.iter_pairs()), [('revision', 'mbp@sourcefrog.net-123-abc'), ('timestamp', '1130653962'), ('timezone', '36000'), ('committer', 'Martin Pool <mbp@test.sourcefrog.net>')])
    self.assertEqual(len(s), 4)