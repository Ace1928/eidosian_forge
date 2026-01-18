import fixtures
from unittest import mock
from oslo_config import cfg
from stevedore import driver
import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
from oslo_messaging import transport
def test_set_default_control_exchange(self):
    oslo_messaging.set_transport_defaults(control_exchange='foo')
    driver.DriverManager = mock.Mock()
    invoke_kwds = dict(default_exchange='foo', allowed_remote_exmods=[])
    driver.DriverManager.return_value = _FakeManager(_FakeDriver(self.conf))
    oslo_messaging.get_transport(self.conf)
    driver.DriverManager.assert_called_once_with(mock.ANY, mock.ANY, invoke_on_load=mock.ANY, invoke_args=mock.ANY, invoke_kwds=invoke_kwds)