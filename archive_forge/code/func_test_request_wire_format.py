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
def test_request_wire_format(self):
    self.conf.oslo_messaging_rabbit.kombu_compression = self.compression
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    driver = transport._driver
    target = oslo_messaging.Target(topic=self.topic, server=self.server, fanout=self.fanout)
    connection, channel, queue = _declare_queue(target)
    self.addCleanup(connection.release)
    driver.send(target, self.ctxt, self.msg)
    msgs = []

    def callback(msg):
        msg = channel.message_to_python(msg)
        msg.ack()
        msgs.append(msg.payload)
    queue.consume(callback=callback, consumer_tag='1', nowait=False)
    connection.drain_events()
    self.assertEqual(1, len(msgs))
    self.assertIn('oslo.message', msgs[0])
    received = msgs[0]
    received['oslo.message'] = jsonutils.loads(received['oslo.message'])
    expected_msg = {'_unique_id': self.uuids[0].hex}
    expected_msg.update(self.expected)
    expected_msg.update(self.expected_ctxt)
    expected = {'oslo.version': '2.0', 'oslo.message': expected_msg}
    self.assertEqual(expected, received)