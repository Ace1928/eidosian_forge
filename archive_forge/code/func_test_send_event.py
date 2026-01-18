import json
from unittest import mock
import ddt
from osprofiler.drivers import loginsight
from osprofiler import exc
from osprofiler.tests import test
@mock.patch.object(loginsight.LogInsightClient, '_send_request')
def test_send_event(self, send_request):
    event = mock.sentinel.event
    self._client.send_event(event)
    exp_body = {'events': [event]}
    exp_path = 'api/v1/events/ingest/%s' % self._client.LI_OSPROFILER_AGENT_ID
    send_request.assert_called_once_with('post', 'http', exp_path, body=exp_body)