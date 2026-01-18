import json
from unittest import mock
import uuid
import requests
from cinderclient import client
from cinderclient import exceptions
from cinderclient.tests.unit import utils
@mock.patch.object(requests, 'request', mock_request)
def test_post_call():
    cl.post('/hi', body=[1, 2, 3])
    headers = {'X-Auth-Token': 'token', 'X-Auth-Project-Id': 'project_id', 'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': cl.USER_AGENT}
    mock_request.assert_called_with('POST', 'http://example.com/hi', headers=headers, data='[1, 2, 3]', **self.TEST_REQUEST_BASE)