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
class TestRabbitQos(test_utils.BaseTestCase):

    def connection_with(self, prefetch, purpose):
        self.config(rabbit_qos_prefetch_count=prefetch, group='oslo_messaging_rabbit')
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        transport._driver._get_connection(purpose)

    @mock.patch('kombu.transport.memory.Channel.basic_qos')
    def test_qos_sent_on_listen_connection(self, fake_basic_qos):
        self.connection_with(prefetch=1, purpose=driver_common.PURPOSE_LISTEN)
        fake_basic_qos.assert_called_once_with(0, 1, False)

    @mock.patch('kombu.transport.memory.Channel.basic_qos')
    def test_qos_not_sent_when_cfg_zero(self, fake_basic_qos):
        self.connection_with(prefetch=0, purpose=driver_common.PURPOSE_LISTEN)
        fake_basic_qos.assert_not_called()

    @mock.patch('kombu.transport.memory.Channel.basic_qos')
    def test_qos_not_sent_on_send_connection(self, fake_basic_qos):
        self.connection_with(prefetch=1, purpose=driver_common.PURPOSE_SEND)
        fake_basic_qos.assert_not_called()