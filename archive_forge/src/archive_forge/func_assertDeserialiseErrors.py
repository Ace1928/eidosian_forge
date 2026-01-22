from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def assertDeserialiseErrors(self, text):
    self.assertRaises((ValueError, IndexError), self.module._deserialise_internal_node, text, stuple(b'not-a-real-sha'))