import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_post_redirect(self):
    """
        Test POST with 302 redirect
        """
    response = self.client.do_request('POST', '/302')
    self.assertEqual(http.OK, response.status)
    self.assertEqual(b'success_from_host_one', response.read())