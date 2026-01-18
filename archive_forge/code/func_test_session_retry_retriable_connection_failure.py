from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
def test_session_retry_retriable_connection_failure(self):
    ok_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, b'OK', http_client.OK)
    fake_session = utils.mockSession({})
    fake_session.request.side_effect = iter((kexc.RetriableConnectionFailure(), ok_resp))
    client = _session_client(session=fake_session)
    client.json_request('GET', '/v1/resources')
    self.assertEqual(2, fake_session.request.call_count)