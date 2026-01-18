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
def test_sender_credit_blocked(self):
    self._blocked_links = set()

    def _on_active(link):
        if self._broker._addresser._is_multicast(link.source_address):
            self._blocked_links.add(link)
        else:
            link.add_capacity(10)
            for li in self._blocked_links:
                li.add_capacity(10)
    self._broker.on_receiver_active = _on_active
    self._broker.on_credit_exhausted = lambda link: None
    self._broker.start()
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic', server='server')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 4)
    target.fanout = True
    target.server = None
    for i in range(3):
        t = threading.Thread(target=driver.send, args=(target, {'context': 'whatever'}, {'msg': 'n=%d' % i}), kwargs={'wait_for_reply': False})
        t.start()
        t.join(timeout=30)
    time.sleep(1)
    self.assertEqual(self._broker.fanout_sent_count, 0)
    target.fanout = False
    rc = driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'id': 'e1'}, wait_for_reply=True)
    self.assertIsNotNone(rc)
    self.assertEqual(rc.get('correlation-id'), 'e1')
    listener.join(timeout=30)
    self.assertTrue(self._broker.fanout_count == 3)
    self.assertFalse(listener.is_alive())
    driver.cleanup()