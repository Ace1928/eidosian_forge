import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertFailUnhexlify(self, as_hex):
    self.assertIs(None, self.module._py_unhexlify(as_hex))