import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def test_request_url_v2_0(self):
    request = webob.Request.blank('/v2.0/images')
    self.middleware.process_request(request)
    self.assertEqual('/v2/images', request.path_info)