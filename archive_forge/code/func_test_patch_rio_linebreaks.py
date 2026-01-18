import re
from tempfile import TemporaryFile
from breezy.tests import TestCase
from .. import rio
from ..rio import RioReader, Stanza, read_stanza, read_stanzas, rio_file
def test_patch_rio_linebreaks(self):
    stanza = Stanza(breaktest='linebreak -/' * 30)
    self.assertContainsRe(rio.to_patch_lines(stanza, 71)[0], b'linebreak\\\\\n')
    stanza = Stanza(breaktest='linebreak-/' * 30)
    self.assertContainsRe(rio.to_patch_lines(stanza, 70)[0], b'linebreak-\\\\\n')
    stanza = Stanza(breaktest='linebreak/' * 30)
    self.assertContainsRe(rio.to_patch_lines(stanza, 70)[0], b'linebreak\\\\\n')