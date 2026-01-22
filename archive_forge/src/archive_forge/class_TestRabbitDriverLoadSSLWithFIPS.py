import datetime
import ssl
import sys
import threading
import time
import uuid
import fixtures
import kombu
import kombu.connection
import kombu.transport.memory
from oslo_serialization import jsonutils
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import common as driver_common
from oslo_messaging._drivers import impl_rabbit as rabbit_driver
from oslo_messaging.exceptions import ConfigurationError
from oslo_messaging.exceptions import MessageDeliveryFailure
from oslo_messaging.tests import utils as test_utils
from oslo_messaging.transport import DriverLoadFailure
from unittest import mock
class TestRabbitDriverLoadSSLWithFIPS(test_utils.BaseTestCase):
    scenarios = [('ssl_fips_mode', dict(options=dict(ssl=True, ssl_enforce_fips_mode=True), expected=True))]

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('kombu.connection.Connection')
    def test_driver_load_with_fips_supported(self, connection_klass, fake_ensure):
        self.config(ssl=True, ssl_enforce_fips_mode=True, group='oslo_messaging_rabbit')
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        with mock.patch.object(ssl, 'FIPS_mode', create=True, return_value=True):
            with mock.patch.object(ssl, 'FIPS_mode_set', create=True):
                connection = transport._driver._get_connection()
                connection_klass.assert_called_once_with('memory:///', transport_options={'client_properties': {'capabilities': {'connection.blocked': True, 'consumer_cancel_notify': True, 'authentication_failure_close': True}, 'connection_name': connection.name}, 'confirm_publish': True, 'on_blocked': mock.ANY, 'on_unblocked': mock.ANY}, ssl=self.expected, login_method='AMQPLAIN', heartbeat=60, failover_strategy='round-robin')

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('oslo_messaging._drivers.impl_rabbit.ssl')
    @mock.patch('kombu.connection.Connection')
    def test_fips_unsupported(self, connection_klass, fake_ssl, fake_ensure):
        self.config(ssl=True, ssl_enforce_fips_mode=True, group='oslo_messaging_rabbit')
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        del fake_ssl.FIPS_mode
        self.assertRaises(ConfigurationError, transport._driver._get_connection)