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
def _gen_brokers(self):
    s2_conf = self._ssl_config.copy()
    for item in ['name', 'key', 'req', 'cert']:
        s2_conf['s_%s' % item] = s2_conf['s2_%s' % item]
    return [FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config), FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=s2_conf['s_name'], ssl_config=s2_conf)]