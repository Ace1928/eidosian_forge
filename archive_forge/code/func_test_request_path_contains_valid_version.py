import webob
from heat.api.middleware import version_negotiation as vn
from heat.tests import common
def test_request_path_contains_valid_version(self):
    version_negotiation = vn.VersionNegotiationFilter(self._version_controller_factory, None, None)
    major_version = 1
    minor_version = 0
    request = webob.Request({'PATH_INFO': 'v{0}.{1}/resource'.format(major_version, minor_version)})
    response = version_negotiation.process_request(request)
    self.assertIsNone(response)
    self.assertEqual(major_version, request.environ['api.major_version'])
    self.assertEqual(minor_version, request.environ['api.minor_version'])