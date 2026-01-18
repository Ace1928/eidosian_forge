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
def test_version_data_microversions(self):
    """Validate [min_|max_]version conversion to {min|max}_microversion."""

    def setup_mock(versions_in):
        jsondata = {'versions': [dict({'status': discover.Status.CURRENT, 'id': 'v2.2', 'links': [{'href': V3_URL, 'rel': 'self'}]}, **versions_in)]}
        self.requests_mock.get(V3_URL, status_code=200, json=jsondata)

    def test_ok(versions_in, versions_out):
        setup_mock(versions_in)
        self.assertEqual([dict({'collection': None, 'version': (2, 2), 'url': V3_URL, 'status': discover.Status.CURRENT, 'raw_status': discover.Status.CURRENT}, **versions_out)], discover.Discover(self.session, V3_URL).version_data())

    def test_exc(versions_in):
        setup_mock(versions_in)
        self.assertRaises(TypeError, discover.Discover(self.session, V3_URL).version_data)
    test_ok({}, {'min_microversion': None, 'max_microversion': None, 'next_min_version': None, 'not_before': None})
    test_ok({'version': '2.2'}, {'min_microversion': None, 'max_microversion': (2, 2), 'next_min_version': None, 'not_before': None})
    test_ok({'min_version': '2', 'version': 'foo', 'max_version': '2.2'}, {'min_microversion': (2, 0), 'max_microversion': (2, 2), 'next_min_version': None, 'not_before': None})
    test_ok({'min_version': '', 'version': '2.1', 'max_version': ''}, {'min_microversion': None, 'max_microversion': (2, 1), 'next_min_version': None, 'not_before': None})
    test_ok({'min_version': '2', 'max_version': '2.2', 'next_min_version': '2.1', 'not_before': '2019-07-01'}, {'min_microversion': (2, 0), 'max_microversion': (2, 2), 'next_min_version': (2, 1), 'not_before': '2019-07-01'})
    test_exc({'min_version': 'foo', 'max_version': '2.1'})
    test_exc({'min_version': '2.1', 'max_version': 'foo'})
    test_exc({'min_version': '2.1', 'version': 'foo'})
    test_exc({'next_min_version': 'bogus', 'not_before': '2019-07-01'})