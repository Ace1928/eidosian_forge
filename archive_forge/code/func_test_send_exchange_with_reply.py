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
def test_send_exchange_with_reply(self):
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target1 = oslo_messaging.Target(topic='test-topic', exchange='e1')
    listener1 = _ListenerThread(driver.listen(target1, None, None)._poll_style_listener, 1)
    target2 = oslo_messaging.Target(topic='test-topic', exchange='e2')
    listener2 = _ListenerThread(driver.listen(target2, None, None)._poll_style_listener, 1)
    rc = driver.send(target1, {'context': 'whatever'}, {'method': 'echo', 'id': 'e1'}, wait_for_reply=True, timeout=30)
    self.assertIsNotNone(rc)
    self.assertEqual('e1', rc.get('correlation-id'))
    rc = driver.send(target2, {'context': 'whatever'}, {'method': 'echo', 'id': 'e2'}, wait_for_reply=True, timeout=30)
    self.assertIsNotNone(rc)
    self.assertEqual('e2', rc.get('correlation-id'))
    listener1.join(timeout=30)
    self.assertFalse(listener1.is_alive())
    listener2.join(timeout=30)
    self.assertFalse(listener2.is_alive())
    driver.cleanup()