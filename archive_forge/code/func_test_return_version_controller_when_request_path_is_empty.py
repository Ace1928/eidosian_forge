import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_return_version_controller_when_request_path_is_empty(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    request = webob.Request({'PATH_INFO': '/'})
    response = version_negotiation.process_request(request)
    self.assertIsInstance(response, VersionController)