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
def _authentication_test(self, addr, retry=None):
    url = oslo_messaging.TransportURL.parse(self.conf, addr)
    driver = amqp_driver.ProtonDriver(self.conf, url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    try:
        rc = driver.send(target, {'context': True}, {'method': 'echo'}, wait_for_reply=True, retry=retry)
        self.assertIsNotNone(rc)
        listener.join(timeout=30)
        self.assertFalse(listener.is_alive())
    finally:
        driver.cleanup()