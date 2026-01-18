import io
from requests_mock.contrib import fixture
import testtools
from barbicanclient import barbican as barb
from barbicanclient.barbican import Barbican
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.tests import keystone_client_fixtures
def test_should_fail_if_provide_unsupported_api_version(self):
    auth_args = '--no-auth --endpoint http://localhost:9311/ --os-project-id project1'
    endpoint_filter_args = '--interface private --service-type custom-type --service-name Burrbican --region-name RegionTwo --barbican-api-version v2'
    args = auth_args + ' ' + endpoint_filter_args
    argv, remainder = self.parser.parse_known_args(args.split())
    self.assertRaises(exceptions.UnsupportedVersion, self.barbican.create_client, argv)