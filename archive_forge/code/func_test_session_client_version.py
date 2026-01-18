from unittest import mock
import keystoneauth1.exceptions.http as ks_exceptions
import osc_lib.exceptions as exceptions
import oslotest.base as base
import requests
import simplejson as json
from osc_placement import http
from osc_placement import version
from oslo_serialization import jsonutils
def test_session_client_version(self):
    session = mock.Mock()
    ks_filter = {'service_type': 'placement', 'region_name': 'mock_region', 'interface': 'mock_interface'}
    target_version = '1.23'
    client = http.SessionClient(session, ks_filter, api_version=target_version)
    self.assertEqual(client.api_version, target_version)
    session.request.assert_not_called()
    target_version = '1'
    session.request.return_value = FakeResponse(200)
    client = http.SessionClient(session, ks_filter, api_version=target_version)
    self.assertEqual(client.api_version, version.MAX_VERSION_NO_GAP)
    expected_version = 'placement ' + version.MAX_VERSION_NO_GAP
    expected_headers = {'OpenStack-API-Version': expected_version, 'Accept': 'application/json'}
    session.request.assert_called_once_with('/', 'GET', endpoint_filter=ks_filter, headers=expected_headers, raise_exc=False)
    session.reset_mock()
    mock_server_version = '1.10'
    json_mock = {'errors': [{'status': 406, 'title': 'Not Acceptable', 'min_version': '1.0', 'max_version': mock_server_version}]}
    session.request.return_value = FakeResponse(406, content=jsonutils.dump_as_bytes(json_mock))
    client = http.SessionClient(session, ks_filter, api_version=target_version)
    self.assertEqual(client.api_version, mock_server_version)
    session.request.assert_called_once_with('/', 'GET', endpoint_filter=ks_filter, headers=expected_headers, raise_exc=False)