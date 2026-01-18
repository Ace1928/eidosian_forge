import http.client as http
from oslo_utils import encodeutils
from glance.common import exception
from glance.tests import utils as test_utils
def test_specified_error_msg(self):
    msg = exception.GlanceException('test')
    self.assertIn('test', encodeutils.exception_to_unicode(msg))