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
def test_send_retry_fail(self):
    self.assertRaises(oslo_messaging.MessageDeliveryFailure, self._send_retry, reject=3, retries=2)