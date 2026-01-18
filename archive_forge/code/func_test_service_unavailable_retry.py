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
def test_service_unavailable_retry(self):
    """
        Test service unavailable response with retry
        """
    self._do_test_exception('/service-unavailable-retry', exception.ServiceUnavailable)