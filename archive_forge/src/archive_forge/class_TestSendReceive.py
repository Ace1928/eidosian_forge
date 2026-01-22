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
class TestSendReceive(test_utils.BaseTestCase):
    _n_senders = [('single_sender', dict(n_senders=1)), ('multiple_senders', dict(n_senders=10))]
    _context = [('empty_context', dict(ctxt={})), ('with_context', dict(ctxt={'user': 'mark'}))]
    _reply = [('rx_id', dict(rx_id=True, reply=None)), ('none', dict(rx_id=False, reply=None)), ('empty_list', dict(rx_id=False, reply=[])), ('empty_dict', dict(rx_id=False, reply={})), ('false', dict(rx_id=False, reply=False)), ('zero', dict(rx_id=False, reply=0))]
    _failure = [('success', dict(failure=False)), ('failure', dict(failure=True, expected=False)), ('expected_failure', dict(failure=True, expected=True))]
    _timeout = [('no_timeout', dict(timeout=None, call_monitor_timeout=None)), ('timeout', dict(timeout=0.01, call_monitor_timeout=None)), ('call_monitor_timeout', dict(timeout=0.01, call_monitor_timeout=0.02))]

    @classmethod
    def generate_scenarios(cls):
        cls.scenarios = testscenarios.multiply_scenarios(cls._n_senders, cls._context, cls._reply, cls._failure, cls._timeout)

    def test_send_receive(self):
        self.config(kombu_missing_consumer_retry_timeout=0.5, group='oslo_messaging_rabbit')
        self.config(heartbeat_timeout_threshold=0, group='oslo_messaging_rabbit')
        transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
        self.addCleanup(transport.cleanup)
        driver = transport._driver
        target = oslo_messaging.Target(topic='testtopic')
        listener = driver.listen(target, None, None)._poll_style_listener
        senders = []
        replies = []
        msgs = []
        self.ctxt['client_timeout'] = self.call_monitor_timeout

        def send_and_wait_for_reply(i):
            try:
                timeout = self.timeout
                cm_timeout = self.call_monitor_timeout
                replies.append(driver.send(target, self.ctxt, {'tx_id': i}, wait_for_reply=True, timeout=timeout, call_monitor_timeout=cm_timeout))
                self.assertFalse(self.failure)
                self.assertIsNone(self.timeout)
            except (ZeroDivisionError, oslo_messaging.MessagingTimeout) as e:
                replies.append(e)
                self.assertTrue(self.failure or self.timeout is not None)
        while len(senders) < self.n_senders:
            senders.append(threading.Thread(target=send_and_wait_for_reply, args=(len(senders),)))
        for i in range(len(senders)):
            senders[i].start()
            received = listener.poll()[0]
            self.assertIsNotNone(received)
            self.assertEqual(self.ctxt, received.ctxt)
            self.assertEqual({'tx_id': i}, received.message)
            msgs.append(received)
        order = list(range(len(senders) - 1, -1, -1))
        if len(order) > 1:
            order[-1], order[-2] = (order[-2], order[-1])
        for i in order:
            if self.timeout is None:
                if self.failure:
                    try:
                        raise ZeroDivisionError
                    except Exception:
                        failure = sys.exc_info()
                    msgs[i].reply(failure=failure)
                elif self.rx_id:
                    msgs[i].reply({'rx_id': i})
                else:
                    msgs[i].reply(self.reply)
            senders[i].join()
        self.assertEqual(len(senders), len(replies))
        for i, reply in enumerate(replies):
            if self.timeout is not None:
                self.assertIsInstance(reply, oslo_messaging.MessagingTimeout)
            elif self.failure:
                self.assertIsInstance(reply, ZeroDivisionError)
            elif self.rx_id:
                self.assertEqual({'rx_id': order[i]}, reply)
            else:
                self.assertEqual(self.reply, reply)