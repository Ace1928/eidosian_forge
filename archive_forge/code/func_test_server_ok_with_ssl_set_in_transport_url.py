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
@mock.patch('ssl.get_default_verify_paths')
def test_server_ok_with_ssl_set_in_transport_url(self, mock_verify_paths):
    self._broker = FakeBroker(self.conf.oslo_messaging_amqp, sock_addr=self._ssl_config['s_name'], ssl_config=self._ssl_config)
    url = oslo_messaging.TransportURL.parse(self.conf, 'amqp://%s:%d?ssl=1' % (self._broker.host, self._broker.port))
    self._broker.start()
    mock_verify_paths.return_value = mock.Mock(cafile=self._ssl_config['ca_cert'])
    driver = amqp_driver.ProtonDriver(self.conf, url)
    target = oslo_messaging.Target(topic='test-topic')
    listener = _ListenerThread(driver.listen(target, None, None)._poll_style_listener, 1)
    driver.send(target, {'context': 'whatever'}, {'method': 'echo', 'a': 'b'}, wait_for_reply=True, timeout=30)
    listener.join(timeout=30)
    self.assertFalse(listener.is_alive())
    driver.cleanup()