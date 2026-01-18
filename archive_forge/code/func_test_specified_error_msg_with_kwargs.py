import http.client as http
from oslo_utils import encodeutils
from glance.common import exception
from glance.tests import utils as test_utils
def test_specified_error_msg_with_kwargs(self):
    msg = exception.GlanceException('test: %(code)s', code=int(http.INTERNAL_SERVER_ERROR))
    self.assertIn('test: 500', encodeutils.exception_to_unicode(msg))