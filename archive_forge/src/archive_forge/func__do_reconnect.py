import abc
import collections
import logging
import os
import platform
import queue
import random
import sys
import threading
import time
import uuid
from oslo_utils import eventletutils
import proton
import pyngus
from oslo_messaging._drivers.amqp1_driver.addressing import AddresserFactory
from oslo_messaging._drivers.amqp1_driver.addressing import keyify
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_NOTIFY
from oslo_messaging._drivers.amqp1_driver.addressing import SERVICE_RPC
from oslo_messaging._drivers.amqp1_driver import eventloop
from oslo_messaging import exceptions
from oslo_messaging.target import Target
from oslo_messaging import transport
def _do_reconnect(self, reason):
    """Invoked on connection/socket failure, failover and re-connect to the
        messaging service.
        """
    self._reconnecting = False
    if not self._closing:
        host = self.hosts.next()
        LOG.info('Reconnecting to: %(hostname)s:%(port)s', {'hostname': host.hostname, 'port': host.port})
        self.processor.wakeup(lambda: self._do_connect())