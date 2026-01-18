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
def test_sender_minimal_credit(self):
    self.config(reply_link_credit=1, rpc_server_credit=1, group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='test-topic', server='server')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 4)
    for i in range(4):
        threading.Thread(target=driver.send, args=(target, {'context': 'whatever'}, {'method': 'echo'}), kwargs={'wait_for_reply': True}).start()
    predicate = lambda: self._broker.direct_count == 8
    _wait_until(predicate, 30)
    self.assertTrue(predicate())
    listener.join(timeout=30)
    driver.cleanup()