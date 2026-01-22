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
class _AmqpBrokerTestCase(test_utils.BaseTestCase):
    """Creates a single FakeBroker for use by the tests"""

    @testtools.skipUnless(pyngus, 'proton modules not present')
    def setUp(self):
        super(_AmqpBrokerTestCase, self).setUp()
        self._broker = FakeBroker(self.conf.oslo_messaging_amqp)
        self._broker_addr = 'amqp://%s:%d' % (self._broker.host, self._broker.port)
        self._broker_url = oslo_messaging.TransportURL.parse(self.conf, self._broker_addr)

    def tearDown(self):
        super(_AmqpBrokerTestCase, self).tearDown()
        if self._broker:
            self._broker.stop()