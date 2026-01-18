import http.client as http
import eventlet.patcher
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_redirect_to_new_host(self):
    """
        Test redirect to one host and then another.
        """
    url = '/redirect-to-%d' % self.port_two
    response = self.client.do_request('POST', url)
    self.assertEqual(http.OK, response.status)
    self.assertEqual(b'success_from_host_two', response.read())
    response = self.client.do_request('POST', '/success')
    self.assertEqual(http.OK, response.status)
    self.assertEqual(b'success_from_host_one', response.read())