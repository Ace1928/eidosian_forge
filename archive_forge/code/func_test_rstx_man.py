from io import StringIO
import breezy.commands
from . import TestCase
def test_rstx_man(self):
    from breezy.doc_generate import autodoc_rstx
    autodoc_rstx.infogen(self.options, self.sio)
    self.assertNotEqual('', self.sio.getvalue())