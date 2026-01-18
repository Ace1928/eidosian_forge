import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_to_lines(self):
    """Write simple rio stanza to string"""
    s = Stanza(number='42', name='fred')
    self.assertEqual(list(s.to_lines()), [b'name: fred\n', b'number: 42\n'])