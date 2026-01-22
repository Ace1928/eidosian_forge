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
class TestRabbitTransportURL(test_utils.BaseTestCase):
    scenarios = [('none', dict(url=None, expected=['amqp://guest:guest@localhost:5672/'])), ('memory', dict(url='kombu+memory:////', expected=['memory:///'])), ('empty', dict(url='rabbit:///', expected=['amqp://guest:guest@localhost:5672/'])), ('localhost', dict(url='rabbit://localhost/', expected=['amqp://:@localhost:5672/'])), ('virtual_host', dict(url='rabbit:///vhost', expected=['amqp://guest:guest@localhost:5672/vhost'])), ('no_creds', dict(url='rabbit://host/virtual_host', expected=['amqp://:@host:5672/virtual_host'])), ('no_port', dict(url='rabbit://user:password@host/virtual_host', expected=['amqp://user:password@host:5672/virtual_host'])), ('full_url', dict(url='rabbit://user:password@host:10/virtual_host', expected=['amqp://user:password@host:10/virtual_host'])), ('full_two_url', dict(url='rabbit://user:password@host:10,user2:password2@host2:12/virtual_host', expected=['amqp://user:password@host:10/virtual_host', 'amqp://user2:password2@host2:12/virtual_host'])), ('rabbit_ipv6', dict(url='rabbit://u:p@[fd00:beef:dead:55::133]:10/vhost', expected=['amqp://u:p@[fd00:beef:dead:55::133]:10/vhost'])), ('rabbit_ipv4', dict(url='rabbit://user:password@10.20.30.40:10/vhost', expected=['amqp://user:password@10.20.30.40:10/vhost'])), ('rabbit_no_vhost_slash', dict(url='rabbit://user:password@10.20.30.40:10', expected=['amqp://user:password@10.20.30.40:10/']))]

    def setUp(self):
        super(TestRabbitTransportURL, self).setUp()
        self.messaging_conf.transport_url = 'rabbit:/'
        self.config(heartbeat_timeout_threshold=0, group='oslo_messaging_rabbit')

    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
    @mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.reset')
    def test_transport_url(self, fake_reset, fake_ensure):
        transport = oslo_messaging.get_transport(self.conf, self.url)
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        urls = driver._get_connection()._url.split(';')
        self.assertEqual(sorted(self.expected), sorted(urls))