import fixtures
import threading
from oslo_config import cfg
import testscenarios
import oslo_messaging
from oslo_messaging.notify import dispatcher
from oslo_messaging.notify import notifier as msg_notifier
from oslo_messaging.tests import utils as test_utils
from unittest import mock
class TestListenerTransportWarning(test_utils.BaseTestCase):

    @mock.patch('oslo_messaging.notify.listener.LOG')
    def test_warning_when_rpc_transport(self, log):
        transport = oslo_messaging.get_rpc_transport(self.conf)
        target = oslo_messaging.Target(topic='foo')
        endpoints = [object()]
        oslo_messaging.get_notification_listener(transport, [target], endpoints)
        log.warning.assert_called_once_with('Using RPC transport for notifications. Please use get_notification_transport to obtain a notification transport instance.')