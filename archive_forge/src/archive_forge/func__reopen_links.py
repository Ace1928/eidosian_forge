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
def _reopen_links(self):
    LOG.debug('Server subscription reopening')
    self._reopen_scheduled = False
    if self._connection:
        for i in range(len(self._receivers)):
            link = self._receivers[i]
            if link.closed:
                addr = link.target_address
                name = link.name
                link.destroy()
                self._receivers[i] = self._open_link(addr, name)