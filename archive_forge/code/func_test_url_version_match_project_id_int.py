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
def test_url_version_match_project_id_int(self):
    self.session = session.Session()
    discovery_fixture = fixture.V3Discovery(V3_URL)
    discovery_doc = _create_single_version(discovery_fixture)
    self.requests_mock.get(V3_URL, status_code=200, json=discovery_doc)
    epd = discover.EndpointData(catalog_url=V3_URL).get_versioned_data(session=self.session, project_id='3')
    self.assertEqual(epd.catalog_url, epd.url)