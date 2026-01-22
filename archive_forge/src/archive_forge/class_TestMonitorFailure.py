import testscenarios
import time
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging import serializer as msg_serializer
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestMonitorFailure(test_utils.BaseTestCase):
    """Test what happens when the call monitor watchdog hits an exception when
    sending the heartbeat.
    """

    class _SleepyEndpoint(object):

        def __init__(self, target=None):
            self.target = target

        def sleep(self, ctxt, **kwargs):
            time.sleep(kwargs['timeout'])
            return True

    def test_heartbeat_failure(self):
        endpoints = [self._SleepyEndpoint()]
        dispatcher = oslo_messaging.RPCDispatcher(endpoints, serializer=None)
        message = {'method': 'sleep', 'args': {'timeout': 3.5}}
        ctxt = {'test': 'value'}
        incoming = mock.Mock(ctxt=ctxt, message=message, client_timeout=2.0)
        incoming.heartbeat = mock.Mock(side_effect=Exception('BOOM!'))
        res = dispatcher.dispatch(incoming)
        self.assertTrue(res)
        self.assertEqual(1, incoming.heartbeat.call_count)