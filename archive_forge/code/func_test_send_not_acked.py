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
def test_send_not_acked(self):
    """Verify exception thrown ack dropped."""
    self.config(pre_settled=[], group='oslo_messaging_amqp')
    driver = amqp_driver.ProtonDriver(self.conf, self._broker_url)
    driver._default_send_timeout = 2
    target = oslo_messaging.Target(topic='!no-ack!')
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, driver.send, target, {'context': 'whatever'}, {'method': 'drop'}, retry=0, wait_for_reply=True)
    driver.cleanup()