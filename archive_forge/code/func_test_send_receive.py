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
def test_send_receive(self):
    transport = oslo_messaging.get_transport(self.conf, 'kombu+memory:////')
    self.addCleanup(transport.cleanup)
    driver = transport._driver
    target = oslo_messaging.Target(topic='testtopic')
    listener = driver.listen(target, None, None)._poll_style_listener
    senders = []
    replies = []
    msgs = []
    wait_conditions = []
    orig_reply_waiter = amqpdriver.ReplyWaiter.wait

    def reply_waiter(self, msg_id, timeout, call_monitor_timeout, reply_q):
        if wait_conditions:
            cond = wait_conditions.pop()
            with cond:
                cond.notify()
            with cond:
                cond.wait()
        return orig_reply_waiter(self, msg_id, timeout, call_monitor_timeout, reply_q)
    self.useFixture(fixtures.MockPatchObject(amqpdriver.ReplyWaiter, 'wait', reply_waiter))

    def send_and_wait_for_reply(i, wait_for_reply):
        replies.append(driver.send(target, {}, {'tx_id': i}, wait_for_reply=wait_for_reply, timeout=None))
    while len(senders) < 2:
        t = threading.Thread(target=send_and_wait_for_reply, args=(len(senders), True))
        t.daemon = True
        senders.append(t)
    t = threading.Thread(target=send_and_wait_for_reply, args=(len(senders), False))
    t.daemon = True
    senders.append(t)
    notify_condition = threading.Condition()
    wait_conditions.append(notify_condition)
    with notify_condition:
        senders[0].start()
        notify_condition.wait()
    msgs.extend(listener.poll())
    self.assertEqual({'tx_id': 0}, msgs[-1].message)
    senders[1].start()
    msgs.extend(listener.poll())
    self.assertEqual({'tx_id': 1}, msgs[-1].message)
    msgs[0].reply({'rx_id': 0})
    msgs[1].reply({'rx_id': 1})
    senders[1].join()
    senders[2].start()
    msgs.extend(listener.poll())
    self.assertEqual({'tx_id': 2}, msgs[-1].message)
    with mock.patch.object(msgs[2], '_send_reply') as method:
        msgs[2].reply({'rx_id': 2})
        self.assertEqual(0, method.call_count)
    senders[2].join()
    with notify_condition:
        notify_condition.notify()
    senders[0].join()
    self.assertEqual(len(senders), len(replies))
    self.assertEqual({'rx_id': 1}, replies[0])
    self.assertIsNone(replies[1])
    self.assertEqual({'rx_id': 0}, replies[2])