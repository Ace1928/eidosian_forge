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
def test_send_no_reply(self):
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    rc = driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
    self.assertIsNone(rc)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    self.assertEqual({'msg': 'value'}, listener.messages.get().message)
    predicate = lambda: self._broker.sender_link_ack_count == 1
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    driver.cleanup()