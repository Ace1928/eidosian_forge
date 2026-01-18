from unittest import mock
from oslo_metrics import message_router
from oslotest import base
import prometheus_client
def test_process_counter(self):
    received_json = '{\n  "module": "oslo_messaging",\n  "name": "rpc_server_invocation_start_total",\n  "action": {\n    "action": "inc",\n    "value": null\n  },\n  "labels": {\n    "exchange": "foo",\n    "topic": "bar",\n    "server": "foobar",\n    "endpoint": "endpoint",\n    "namespace": "ns",\n    "version": "v2",\n    "method": "get",\n    "process": "done"\n  }\n}'.encode()
    with mock.patch.object(prometheus_client.Counter, 'inc') as mock_inc:
        router = message_router.MessageRouter()
        router.process(received_json)
        mock_inc.assert_called_once_with()