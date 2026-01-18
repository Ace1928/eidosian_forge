from io import BytesIO
import breezy.bzr.bzrdir
import breezy.mergeable
import breezy.transport
import breezy.urlutils
from ... import errors, tests
from ...tests.per_transport import transport_test_permutations
from ...tests.scenarios import load_tests_apply_scenarios
from ...tests.test_transport import TestTransportImplementation
from ..bundle.serializer import write_bundle
def test_read_fail(self):
    self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree'))
    self.assertRaises(errors.NotABundle, self.read_mergeable_from_url, self.get_url('tree/a'))