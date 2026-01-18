import json
import logging
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import adapter
from keystoneauth1 import exceptions as keystone_exception
from oslo_serialization import jsonutils
from cinderclient import api_versions
import cinderclient.client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
@mock.patch.object(adapter.Adapter, 'request')
@mock.patch.object(exceptions, 'from_response')
def test_sessionclient_request_method(self, mock_from_resp, mock_request):
    kwargs = {'body': {'volume': {'status': 'creating', 'imageRef': 'username', 'attach_status': 'detached'}, 'authenticated': 'True'}}
    resp = {'text': {'volume': {'status': 'creating', 'id': '431253c0-e203-4da2-88df-60c756942aaf', 'size': 1}}, 'code': 202}
    request_id = 'req-f551871a-4950-4225-9b2c-29a14c8f075e'
    mock_response = utils.TestResponse({'status_code': 202, 'text': json.dumps(resp).encode('latin-1'), 'headers': {'x-openstack-request-id': request_id}})
    mock_request.return_value = mock_response
    session_client = cinderclient.client.SessionClient(session=mock.Mock())
    response, body = session_client.request(mock.sentinel.url, 'POST', **kwargs)
    self.assertIsNotNone(session_client._logger)
    self.assertEqual(202, response.status_code)
    self.assertFalse(mock_from_resp.called)