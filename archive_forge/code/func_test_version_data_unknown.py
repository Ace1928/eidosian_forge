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
def test_version_data_unknown(self):
    discovery_fixture = fixture.V3Discovery(V3_URL)
    discovery_fixture.status = 'hungry'
    discovery_doc = _create_single_version(discovery_fixture)
    self.requests_mock.get(V3_URL, status_code=200, json=discovery_doc)
    disc = discover.Discover(self.session, V3_URL)
    clean_data = disc.version_data(allow_unknown=True)
    self.assertEqual(discover.Status.UNKNOWN, clean_data[0]['status'])