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
def test_released_send(self):
    """Verify exception thrown if send Nacked."""
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    target = oslo_messaging.Target(topic='no listener')
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send, target, {'context': 'whatever'}, {'method': 'drop'}, wait_for_reply=True, retry=0, timeout=1.0)
    driver.cleanup()