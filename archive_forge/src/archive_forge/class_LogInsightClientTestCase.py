import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
class LogInsightClientTestCase(test.TestCase):

    def setUp(self):
        super(LogInsightClientTestCase, self).setUp()
        self._host = 'localhost'
        self._username = 'username'
        self._password = 'password'
        self._client = loginsight.LogInsightClient(self._host, self._username, self._password)
        self._client._session_id = '4ff800d1-3175-4b49-9209-39714ea56416'

    def test_check_response_login_timeout(self):
        resp = mock.Mock(status_code=440)
        self.assertRaises(exc.LogInsightLoginTimeout, self._client._check_response, resp)

    def test_check_response_api_error(self):
        resp = mock.Mock(status_code=401, ok=False)
        resp.text = json.dumps({'errorMessage': 'Invalid username or password.', 'errorCode': 'FIELD_ERROR'})
        e = self.assertRaises(exc.LogInsightAPIError, self._client._check_response, resp)
        self.assertEqual('Invalid username or password.', str(e))

    @mock.patch('requests.Request')
    @mock.patch('json.dumps')
    @mock.patch.object(loginsight.LogInsightClient, '_check_response')
    def test_send_request(self, check_resp, json_dumps, request_class):
        req = mock.Mock()
        request_class.return_value = req
        prep_req = mock.sentinel.prep_req
        req.prepare = mock.Mock(return_value=prep_req)
        data = mock.sentinel.data
        json_dumps.return_value = data
        self._client._session = mock.Mock()
        resp = mock.Mock()
        self._client._session.send = mock.Mock(return_value=resp)
        resp_json = mock.sentinel.resp_json
        resp.json = mock.Mock(return_value=resp_json)
        header = {'X-LI-Session-Id': 'foo'}
        body = mock.sentinel.body
        params = mock.sentinel.params
        ret = self._client._send_request('get', 'https', 'api/v1/events', header, body, params)
        self.assertEqual(resp_json, ret)
        exp_headers = {'X-LI-Session-Id': 'foo', 'content-type': 'application/json'}
        request_class.assert_called_once_with('get', 'https://localhost:9543/api/v1/events', headers=exp_headers, data=data, params=mock.sentinel.params)
        self._client._session.send.assert_called_once_with(prep_req, verify=False)
        check_resp.assert_called_once_with(resp)

    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_is_current_session_active_with_active_session(self, send_request):
        self.assertTrue(self._client._is_current_session_active())
        exp_header = {'X-LI-Session-Id': self._client._session_id}
        send_request.assert_called_once_with('get', 'https', 'api/v1/sessions/current', headers=exp_header)

    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_is_current_session_active_with_expired_session(self, send_request):
        send_request.side_effect = exc.LogInsightLoginTimeout
        self.assertFalse(self._client._is_current_session_active())
        send_request.assert_called_once_with('get', 'https', 'api/v1/sessions/current', headers={'X-LI-Session-Id': self._client._session_id})

    @mock.patch.object(loginsight.LogInsightClient, '_is_current_session_active', return_value=True)
    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_login_with_current_session_active(self, send_request, is_current_session_active):
        self._client.login()
        is_current_session_active.assert_called_once_with()
        send_request.assert_not_called()

    @mock.patch.object(loginsight.LogInsightClient, '_is_current_session_active', return_value=False)
    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_login(self, send_request, is_current_session_active):
        new_session_id = '569a80aa-be5c-49e5-82c1-bb62392d2667'
        resp = {'sessionId': new_session_id}
        send_request.return_value = resp
        self._client.login()
        is_current_session_active.assert_called_once_with()
        exp_body = {'username': self._username, 'password': self._password}
        send_request.assert_called_once_with('post', 'https', 'api/v1/sessions', body=exp_body)
        self.assertEqual(new_session_id, self._client._session_id)

    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_send_event(self, send_request):
        event = mock.sentinel.event
        self._client.send_event(event)
        exp_body = {'events': [event]}
        exp_path = 'api/v1/events/ingest/%s' % self._client.LI_OSPROFILER_AGENT_ID
        send_request.assert_called_once_with('post', 'http', exp_path, body=exp_body)

    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    def test_query_events(self, send_request):
        resp = mock.sentinel.response
        send_request.return_value = resp
        self.assertEqual(resp, self._client.query_events({'foo': 'bar'}))
        exp_header = {'X-LI-Session-Id': self._client._session_id}
        exp_params = {'limit': 20000, 'timeout': self._client._query_timeout}
        send_request.assert_called_once_with('get', 'https', 'api/v1/events/foo/CONTAINS+bar/timestamp/GT+0', headers=exp_header, params=exp_params)

    @mock.patch.object(loginsight.LogInsightClient, '_send_request')
    @mock.patch.object(loginsight.LogInsightClient, 'login')
    def test_query_events_with_session_expiry(self, login, send_request):
        resp = mock.sentinel.response
        send_request.side_effect = [exc.LogInsightLoginTimeout, resp]
        self.assertEqual(resp, self._client.query_events({'foo': 'bar'}))
        login.assert_called_once_with()
        exp_header = {'X-LI-Session-Id': self._client._session_id}
        exp_params = {'limit': 20000, 'timeout': self._client._query_timeout}
        exp_send_request_call = mock.call('get', 'https', 'api/v1/events/foo/CONTAINS+bar/timestamp/GT+0', headers=exp_header, params=exp_params)
        send_request.assert_has_calls([exp_send_request_call] * 2)