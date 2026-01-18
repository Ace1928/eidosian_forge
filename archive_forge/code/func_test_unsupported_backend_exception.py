from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_unsupported_backend_exception(self):
    msg = glance_store.UnsupportedBackend()
    self.assertIn('', encodeutils.exception_to_unicode(msg))