import http.client as http_client
import eventlet.patcher
import httplib2
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def test_server_traceback(self):
    """
        Verify that the wsgi server does not return tracebacks to the client on
        500 errors (bug 1192132)
        """
    http = httplib2.Http()
    path = 'http://%s:%d/server-traceback' % ('127.0.0.1', self.port)
    response, content = http.request(path, 'GET')
    self.assertNotIn(b'ServerError', content)
    self.assertEqual(http_client.INTERNAL_SERVER_ERROR, response.status)