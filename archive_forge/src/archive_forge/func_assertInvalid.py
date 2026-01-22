import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
def assertInvalid(self, data):
    """Ensure that we get a proper error when trying to parse invalid bytes.

        (mostly this is testing that bad input doesn't cause us to segfault)
        """
    self.assertRaises((ValueError, TypeError), self.module._parse_into_chk, data, 1, 0)