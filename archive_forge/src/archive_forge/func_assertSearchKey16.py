from ... import tests
from .. import chk_map
from ..static_tuple import StaticTuple
def assertSearchKey16(self, expected, key):
    self.assertEqual(expected, self.module._search_key_16(key))