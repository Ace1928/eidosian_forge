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
@mock.patch('oslo_messaging._drivers.impl_rabbit.LOG')
@mock.patch('kombu.connection.Connection.heartbeat_check')
@mock.patch('oslo_messaging._drivers.impl_rabbit.Connection._heartbeat_supported_and_enabled', return_value=True)
@mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
def _do_test_heartbeat_sent(self, fake_ensure_connection, fake_heartbeat_support, fake_heartbeat, fake_logger, heartbeat_side_effect=None, info=None):
    event = eventletutils.Event()

    def heartbeat_check(rate=2):
        event.set()
        if heartbeat_side_effect:
            raise heartbeat_side_effect
    fake_heartbeat.side_effect = heartbeat_check
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    conn = transport._driver._get_connection()
    conn.ensure(method=lambda: True)
    event.wait()
    conn._heartbeat_stop()
    self.assertLess(0, fake_heartbeat.call_count)
    if not heartbeat_side_effect:
        self.assertEqual(1, fake_ensure_connection.call_count)
        self.assertEqual(2, fake_logger.debug.call_count)
        self.assertEqual(0, fake_logger.info.call_count)
    else:
        self.assertEqual(2, fake_ensure_connection.call_count)
        self.assertEqual(2, fake_logger.debug.call_count)
        self.assertEqual(1, fake_logger.info.call_count)
        self.assertIn(mock.call(info, mock.ANY), fake_logger.info.mock_calls)