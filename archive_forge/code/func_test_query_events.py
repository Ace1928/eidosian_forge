import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight.LogInsightClient, '_send_request')
def test_query_events(self, send_request):
    resp = mock.sentinel.response
    send_request.return_value = resp
    self.assertEqual(resp, self._client.query_events({'foo': 'bar'}))
    exp_header = {'X-LI-Session-Id': self._client._session_id}
    exp_params = {'limit': 20000, 'timeout': self._client._query_timeout}
    send_request.assert_called_once_with('get', 'https', 'api/v1/events/foo/CONTAINS+bar/timestamp/GT+0', headers=exp_header, params=exp_params)