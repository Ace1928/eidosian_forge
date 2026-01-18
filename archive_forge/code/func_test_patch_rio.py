import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_patch_rio(self):
    stanza = Stanza(data='#\n\r\\r ', space=' ' * 255, hash='#' * 255)
    lines = rio.to_patch_lines(stanza)
    for line in lines:
        self.assertContainsRe(line, b'^# ')
        self.assertTrue(72 >= len(line))
    for line in rio.to_patch_lines(stanza, max_width=12):
        self.assertTrue(12 >= len(line))
    new_stanza = rio.read_patch_stanza(self.mail_munge(lines, dos_nl=False))
    lines = self.mail_munge(lines)
    new_stanza = rio.read_patch_stanza(lines)
    self.assertEqual('#\n\r\\r ', new_stanza.get('data'))
    self.assertEqual(' ' * 255, new_stanza.get('space'))
    self.assertEqual('#' * 255, new_stanza.get('hash'))