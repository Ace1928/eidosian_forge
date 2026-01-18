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
def test_server_ok(self):
    self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config)
    url = 'amqp://%s:%d' % (self._broker.host, self._broker.port)
    self._ssl_server_ok(url)