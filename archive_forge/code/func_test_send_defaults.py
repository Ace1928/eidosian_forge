import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_send_defaults(self):
    t = transport.Transport(_FakeDriver(cfg.CONF))
    t._driver.send = mock.Mock()
    t._send(self._target, 'ctxt', 'message')
    t._driver.send.assert_called_once_with(self._target, 'ctxt', 'message', wait_for_reply=None, timeout=None, call_monitor_timeout=None, retry=None, transport_options=None)