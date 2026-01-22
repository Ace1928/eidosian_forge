from unittest import mock
from zunclient import api_versions
from zunclient import exceptions
from zunclient.tests.unit import utils
from zunclient.v1 import versions
class UpdateHeadersTestCase(utils.TestCase):

    def test_api_version_is_null(self):
        headers = {}
        api_versions.update_headers(headers, api_versions.APIVersion())
        self.assertEqual({}, headers)

    def test_api_version_is_major(self):
        headers = {}
        api_versions.update_headers(headers, api_versions.APIVersion('7.0'))
        self.assertEqual({}, headers)

    def test_api_version_is_not_null(self):
        api_version = api_versions.APIVersion('2.3')
        headers = {}
        api_versions.update_headers(headers, api_version)
        self.assertEqual({'OpenStack-API-Version': 'container %s' % api_version.get_string()}, headers)