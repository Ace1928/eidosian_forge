import http.client as http
import ddt
import webob
from oslo_serialization import jsonutils
from glance.api.middleware import version_negotiation
from glance.api import versions
from glance.tests.unit import base
def test_request_url_v2_8_default_unsupported(self):
    request = webob.Request.blank('/v2.8/images')
    resp = self.middleware.process_request(request)
    self.assertIsInstance(resp, versions.Controller)