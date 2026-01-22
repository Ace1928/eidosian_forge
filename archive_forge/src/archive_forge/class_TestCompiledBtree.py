import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
class TestCompiledBtree(tests.TestCase):

    def test_exists(self):
        self.requireFeature(compiled_btreeparser_feature)