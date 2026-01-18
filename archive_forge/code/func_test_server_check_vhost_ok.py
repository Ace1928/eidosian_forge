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
def test_server_check_vhost_ok(self):
    self.config(ssl_verify_vhost=True, group='oslo_messaging_amqp')
    self._ssl_config['s_cert'] = self._ssl_config['bad_cert']
    self._ssl_config['s_key'] = self._ssl_config['bad_key']
    self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config)
    url = 'amqp://%s:%d/Invalid' % (self._broker.host, self._broker.port)
    self._ssl_server_ok(url)