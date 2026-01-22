import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
@testtools.skipUnless(pyngus, 'proton modules not present')
class TestMessageRetransmit(_AmqpBrokerTestCase):

    def _test_retransmit(self, nack_method):
        self._nack_count = 2

        def _on_message(message, handle, link):
            if self._nack_count:
                self._nack_count -= 1
                nack_method(link, handle)
            else:
                self._broker.forward_message(message, handle, link)
        self._broker.on_message = _on_message
        self._broker.start()
        self.config(link_retry_delay=1, pre_settled=[], group='oslo_messaging_amqp')
        driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
        target = oslo_messaging.Target(topic='test-topic')
        listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
        try:
            rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'blah'}, wait_for_reply=True, retry=2)
        except Exception:
            listener.kill(timeout=30)
            raise
        else:
            self.assertIsNotNone(rc)
            self.assertEqual(0, self._nack_count)
            self.assertEqual(rc.get('correlation-id'), 'blah')
            listener.join(timeout=30)
        finally:
            self.assertFalse(listener.is_alive())
            driver.cleanup()

    def test_released(self):
        self._test_retransmit(lambda link, handle: link.message_released(handle))

    def test_modified(self):
        self._test_retransmit(lambda link, handle: link.message_modified(handle, False, False, {}))

    def test_modified_failed(self):
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self._test_retransmit, lambda link, handle: link.message_modified(handle, True, False, {}))

    def test_rejected(self):
        self.assertRaises(oslo_messaging.MessageDeliveryFailure, self._test_retransmit, lambda link, handle: link.message_rejected(handle, {}))