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
class TestProtonDriverLoad(test_utils.BaseTestCase):

    def setUp(self):
        super(TestProtonDriverLoad, self).setUp()
        self.messaging_conf.transport_url = 'amqp://'

    def test_driver_load(self):
        transport = oslo_messaging.get_transport(self.conf)
        self.assertIsInstance(transport._driver, amqp_driver.ProtonDriver)