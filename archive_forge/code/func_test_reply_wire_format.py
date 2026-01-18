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
@mock.patch('kombu.message.Message.reject')
def test_reply_wire_format(self, reject_mock):
    self.conf.oslo_messaging_rabbit.kombu_compression = None
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    driver = transport._driver
    target = oslo_messaging.Target(topic='testtopic', server=None, fanout=False)
    listener = driver.listen(target, None, None)._poll_style_listener
    connection, producer = _create_producer(target)
    self.addCleanup(connection.release)
    msg = {'oslo.version': '2.0', 'oslo.message': {}}
    msg['oslo.message'].update({'_msg_id': uuid.uuid4().hex, '_unique_id': uuid.uuid4().hex, '_reply_q': 'reply_' + uuid.uuid4().hex, '_timeout': None})
    msg['oslo.message'] = jsonutils.dumps(msg['oslo.message'])
    producer.publish(msg)
    received = listener.poll()[0]
    self.assertIsNotNone(received)
    self.assertEqual({}, received.message)
    producer.publish(msg)
    received = listener.poll(timeout=1)
    self.assertEqual(len(received), 0)
    reject_mock.assert_not_called()