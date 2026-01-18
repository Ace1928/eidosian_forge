import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_version_data_legacy_ironic_microversions(self):
    """Validate detection of legacy Ironic microversion ranges."""
    ironic_url = 'https://bare-metal.example.com/v1/'
    self.requests_mock.get(ironic_url, status_code=200, json={'id': 'v1', 'links': [{'href': ironic_url, 'rel': 'self'}]}, headers={'X-OpenStack-Ironic-API-Minimum-Version': '1.3', 'X-OpenStack-Ironic-API-Maximum-Version': '1.21'})
    self.assertEqual([{'collection': None, 'version': (1, 0), 'url': ironic_url, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT, 'min_microversion': (1, 3), 'max_microversion': (1, 21), 'next_min_version': None, 'not_before': None}], discover.Discover(self.session, ironic_url).version_data())