from oslo_config import cfg
import testscenarios
from unittest import mock
import oslo_messaging
from oslo_messaging import exceptions
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
class TestCallRetry(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(ctor=None, prepare=_notset, expect=None)), ('ctor', dict(ctor=21, prepare=_notset, expect=21)), ('ctor_zero', dict(ctor=0, prepare=_notset, expect=0)), ('prepare', dict(ctor=None, prepare=21, expect=21)), ('prepare_override', dict(ctor=10, prepare=21, expect=21)), ('prepare_zero', dict(ctor=None, prepare=0, expect=0))]

    def test_call_retry(self):
        transport = oslo_messaging.get_rpc_transport(self.conf, url='fake:')
        client = oslo_messaging.get_rpc_client(transport, oslo_messaging.Target(), retry=self.ctor)
        transport._send = mock.Mock()
        msg = dict(method='foo', args={})
        kwargs = dict(wait_for_reply=True, timeout=60, retry=self.expect, call_monitor_timeout=None, transport_options=None)
        if self.prepare is not _notset:
            client = client.prepare(retry=self.prepare)
        client.call({}, 'foo')
        transport._send.assert_called_once_with(oslo_messaging.Target(), {}, msg, **kwargs)