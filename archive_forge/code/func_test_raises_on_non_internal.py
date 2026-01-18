from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def test_raises_on_non_internal(self):
    self.assertDeserialiseErrors(b'')
    self.assertDeserialiseErrors(b'short\n')
    self.assertDeserialiseErrors(b'chknotnode:\n')
    self.assertDeserialiseErrors(b'chknode:x\n')
    self.assertDeserialiseErrors(b'chknode:\n')
    self.assertDeserialiseErrors(b'chknode:\nnotint\n')
    self.assertDeserialiseErrors(b'chknode:\n10\n')
    self.assertDeserialiseErrors(b'chknode:\n10\n256\n')
    self.assertDeserialiseErrors(b'chknode:\n10\n256\n10\n')
    self.assertDeserialiseErrors(b'chknode:\n10\n256\n0\n1\nfo')