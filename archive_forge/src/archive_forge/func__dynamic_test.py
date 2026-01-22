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
def _dynamic_test(self, product):
    broker = FakeBroker(self.conf.oslo_messaging_amqp, product=product)
    broker.start()
    url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d' % (broker.host, broker.port))
    driver = amqp_driver.ProtonDriver(self.conf, url)
    target = oslo_messaging.Target(topic='test-topic', server='Server')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    driver.send(target, {'context': True}, {'msg': 'value'}, wait_for_reply=False)
    listener.join(timeout=30)
    addresser = driver._ctrl.addresser
    driver.cleanup()
    broker.stop()
    return addresser