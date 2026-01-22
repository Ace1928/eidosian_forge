import pprint
import zlib
from ... import errors, fifo_cache, lru_cache, osutils, tests, transport
from ...tests import TestCaseWithTransport, features, scenarios
from .. import btree_index
from .. import index as _mod_index
def assertEqualApproxCompressed(self, expected, actual, slop=6):
    """Check a count of compressed bytes is approximately as expected

        Relying on compressed length being stable even with fixed inputs is
        slightly bogus, but zlib is stable enough that this mostly works.
        """
    if not expected - slop < actual < expected + slop:
        self.fail('Expected around %d bytes compressed but got %d' % (expected, actual))