from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_session_max_retries_none(self):
    error_body = _get_error_body()
    fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.CONFLICT)
    fake_session = utils.mockSession({})
    fake_session.request.return_value = fake_resp
    client = _session_client(session=fake_session)
    client.conflict_max_retries = None
    self.assertRaises(exc.Conflict, client.json_request, 'GET', '/v1/resources')
    self.assertEqual(http.DEFAULT_MAX_RETRIES + 1, fake_session.request.call_count)