import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_repeated_field(self):
    """Repeated field in rio"""
    s = Stanza()
    for k, v in [('a', '10'), ('b', '20'), ('a', '100'), ('b', '200'), ('a', '1000'), ('b', '2000')]:
        s.add(k, v)
    s2 = read_stanza(s.to_lines())
    self.assertEqual(s, s2)
    self.assertEqual(s.get_all('a'), ['10', '100', '1000'])
    self.assertEqual(s.get_all('b'), ['20', '200', '2000'])