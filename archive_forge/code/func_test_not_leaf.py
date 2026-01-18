import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def test_not_leaf(self):
    self.assertInvalid(b'type=internal\n')