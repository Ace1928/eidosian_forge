from oslotest import base as test_base
from oslo_utils import versionutils
def test_requested_minor_less_than(self):
    self.assertTrue(versionutils.is_compatible('1.0', '1.1'))