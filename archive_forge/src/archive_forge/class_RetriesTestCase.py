from http import client as http_client
import json
import time
from unittest import mock
from keystoneauth1 import exceptions as kexc
from ironicclient.common import filecache
from ironicclient.common import http
from ironicclient import exc
from ironicclient.tests.unit import utils
@mock.patch.object(time, 'sleep', lambda *_: None)
class RetriesTestCase(utils.BaseTestCase):

    def test_session_retry(self):
        error_body = _get_error_body()
        fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.CONFLICT)
        ok_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, b'OK', http_client.OK)
        fake_session = utils.mockSession({})
        fake_session.request.side_effect = iter((fake_resp, ok_resp))
        client = _session_client(session=fake_session)
        client.json_request('GET', '/v1/resources')
        self.assertEqual(2, fake_session.request.call_count)

    def test_session_retry_503(self):
        error_body = _get_error_body()
        fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.SERVICE_UNAVAILABLE)
        ok_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, b'OK', http_client.OK)
        fake_session = utils.mockSession({})
        fake_session.request.side_effect = iter((fake_resp, ok_resp))
        client = _session_client(session=fake_session)
        client.json_request('GET', '/v1/resources')
        self.assertEqual(2, fake_session.request.call_count)

    def test_session_retry_connection_refused(self):
        ok_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, b'OK', http_client.OK)
        fake_session = utils.mockSession({})
        fake_session.request.side_effect = iter((exc.ConnectionRefused(), ok_resp))
        client = _session_client(session=fake_session)
        client.json_request('GET', '/v1/resources')
        self.assertEqual(2, fake_session.request.call_count)

    def test_session_retry_retriable_connection_failure(self):
        ok_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, b'OK', http_client.OK)
        fake_session = utils.mockSession({})
        fake_session.request.side_effect = iter((kexc.RetriableConnectionFailure(), ok_resp))
        client = _session_client(session=fake_session)
        client.json_request('GET', '/v1/resources')
        self.assertEqual(2, fake_session.request.call_count)

    def test_session_retry_fail(self):
        error_body = _get_error_body()
        fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.CONFLICT)
        fake_session = utils.mockSession({})
        fake_session.request.return_value = fake_resp
        client = _session_client(session=fake_session)
        self.assertRaises(exc.Conflict, client.json_request, 'GET', '/v1/resources')
        self.assertEqual(http.DEFAULT_MAX_RETRIES + 1, fake_session.request.call_count)

    def test_session_max_retries_none(self):
        error_body = _get_error_body()
        fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.CONFLICT)
        fake_session = utils.mockSession({})
        fake_session.request.return_value = fake_resp
        client = _session_client(session=fake_session)
        client.conflict_max_retries = None
        self.assertRaises(exc.Conflict, client.json_request, 'GET', '/v1/resources')
        self.assertEqual(http.DEFAULT_MAX_RETRIES + 1, fake_session.request.call_count)

    def test_session_change_max_retries(self):
        error_body = _get_error_body()
        fake_resp = utils.mockSessionResponse({'Content-Type': 'application/json'}, error_body, http_client.CONFLICT)
        fake_session = utils.mockSession({})
        fake_session.request.return_value = fake_resp
        client = _session_client(session=fake_session)
        client.conflict_max_retries = http.DEFAULT_MAX_RETRIES + 1
        self.assertRaises(exc.Conflict, client.json_request, 'GET', '/v1/resources')
        self.assertEqual(http.DEFAULT_MAX_RETRIES + 2, fake_session.request.call_count)