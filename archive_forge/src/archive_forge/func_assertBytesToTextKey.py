from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def assertBytesToTextKey(self, key, bytes):
    self.assertEqual(key, self.module._bytes_to_text_key(bytes))