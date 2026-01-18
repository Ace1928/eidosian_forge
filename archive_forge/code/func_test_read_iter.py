import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_read_iter(self):
    """Read several stanzas from file"""
    tmpf = TemporaryFile()
    tmpf.write(b'version_header: 1\n\nname: foo\nval: 123\n\nname: bar\nval: 129319\n')
    tmpf.seek(0)
    reader = read_stanzas(tmpf)
    read_iter = iter(reader)
    stuff = list(reader)
    self.assertEqual(stuff, [Stanza(version_header='1'), Stanza(name='foo', val='123'), Stanza(name='bar', val='129319')])