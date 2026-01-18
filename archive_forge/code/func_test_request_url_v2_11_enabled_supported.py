import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def test_request_url_v2_11_enabled_supported(self):
    self.config(enabled_backends='slow:one,fast:two')
    request = webob.Request.blank('/v2.11/images')
    self.middleware.process_request(request)
    self.assertEqual('/v2/images', request.path_info)