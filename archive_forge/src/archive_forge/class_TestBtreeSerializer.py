import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
class TestBtreeSerializer(tests.TestCase):
    _test_needs_features = [compiled_btreeparser_feature]

    @property
    def module(self):
        return compiled_btreeparser_feature.module