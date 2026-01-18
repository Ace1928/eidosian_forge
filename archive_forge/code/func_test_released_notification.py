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
def test_released_notification(self):
    """Broker sends a Nack (released)"""
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send_notification, oslo_messaging.Target(topic='bad address'), 'context', {'target': 'bad address'}, 2.0, retry=0)
    driver.cleanup()