import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
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