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
class TestReplyWireFormat(test_utils.BaseTestCase):
    _target = [('topic_target', dict(topic='testtopic', server=None, fanout=False)), ('server_target', dict(topic='testtopic', server='testserver', fanout=False)), ('fanout_target', dict(topic='testtopic', server=None, fanout=True))]
    _msg = [('empty_msg', dict(msg={}, expected={})), ('primitive_msg', dict(msg={'foo': 'bar'}, expected={'foo': 'bar'})), ('complex_msg', dict(msg={'a': {'b': '1920-02-03T04:05:06.000007'}}, expected={'a': {'b': '1920-02-03T04:05:06.000007'}}))]
    _context = [('empty_ctxt', dict(ctxt={}, expected_ctxt={'client_timeout': None})), ('user_project_ctxt', dict(ctxt={'_context_user': 'mark', '_context_project': 'snarkybunch'}, expected_ctxt={'user': 'mark', 'project': 'snarkybunch', 'client_timeout': None}))]
    _compression = [('gzip_compression', dict(compression='gzip')), ('without_compression', dict(compression=None))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._msg, cls._context, cls._target, cls._compression)

    def test_reply_wire_format(self):
        self.conf.oslo_messaging_rabbit.kombu_compression = self.compression
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        target = oslo_messaging.Target(topic=self.topic, server=self.server, fanout=self.fanout)
        listener = driver.listen(target, None, None)._poll_style_listener
        connection, producer = _create_producer(target)
        self.addCleanup(connection.release)
        msg = {'oslo.version': '2.0', 'oslo.message': {}}
        msg['oslo.message'].update(self.msg)
        msg['oslo.message'].update(self.ctxt)
        msg['oslo.message'].update({'_msg_id': uuid.uuid4().hex, '_unique_id': uuid.uuid4().hex, '_reply_q': 'reply_' + uuid.uuid4().hex, '_timeout': None})
        msg['oslo.message'] = jsonutils.dumps(msg['oslo.message'])
        producer.publish(msg)
        received = listener.poll()[0]
        self.assertIsNotNone(received)
        self.assertEqual(self.expected_ctxt, received.ctxt)
        self.assertEqual(self.expected, received.message)