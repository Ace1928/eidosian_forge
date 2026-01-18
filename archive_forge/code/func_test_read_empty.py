import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_read_empty(self):
    """Detect end of rio file"""
    s = read_stanza([])
    self.assertEqual(s, None)
    self.assertTrue(s is None)