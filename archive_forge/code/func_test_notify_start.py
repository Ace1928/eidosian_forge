from unittest import mock
from oslo_config import cfg
from osprofiler.drivers import jaeger
from osprofiler import opts
from osprofiler.tests import test
from jaeger_client import Config
@mock.patch('osprofiler._utils.shorten_id')
def test_notify_start(self, mock_shorten_id):
    self.driver.notify(self.payload_start)
    calls = [mock.call(self.payload_start['base_id']), mock.call(self.payload_start['parent_id']), mock.call(self.payload_start['trace_id'])]
    mock_shorten_id.assert_has_calls(calls, any_order=True)