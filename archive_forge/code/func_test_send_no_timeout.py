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
@mock.patch('kombu.messaging.Producer.publish')
def test_send_no_timeout(self, fake_publish):
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    exchange_mock = mock.Mock()
    with transport._driver._get_connection(driver_common.PURPOSE_SEND) as pool_conn:
        conn = pool_conn.connection
        conn._publish(exchange_mock, 'msg', routing_key='routing_key')
    fake_publish.assert_called_with('msg', expiration=None, mandatory=False, compression=self.conf.oslo_messaging_rabbit.kombu_compression, exchange=exchange_mock, routing_key='routing_key')