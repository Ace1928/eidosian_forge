import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_write_empty_stanza(self):
    """Write empty stanza"""
    l = list(Stanza().to_lines())
    self.assertEqual(l, [])