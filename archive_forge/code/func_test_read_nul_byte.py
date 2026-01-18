import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_read_nul_byte(self):
    """File consisting of a nul byte causes an error."""
    self.assertRaises(ValueError, read_stanza, [b'\x00'])