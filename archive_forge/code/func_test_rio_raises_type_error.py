import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_rio_raises_type_error(self):
    """TypeError on adding invalid type to Stanza"""
    s = Stanza()
    self.assertRaises(TypeError, s.add, 'foo', {})