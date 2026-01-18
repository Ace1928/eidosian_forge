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
@mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.ensure_connection')
@mock.patch('oslo_messaging._drivers.impl_rabbit.Connection.reset')
def test_transport_url(self, fake_reset, fake_ensure):
    transport = oslo_messaging.get_transport(self.conf, self.url)
    self.addCleanup(transport.cleanup)
    driver = transport._driver
    urls = driver._get_connection()._url.split(';')
    self.assertEqual(sorted(self.expected), sorted(urls))