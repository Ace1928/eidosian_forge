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
class MessageDispositionTask(Task):
    """A task that updates the message disposition as accepted or released
    for a Server
    """

    def __init__(self, disposition, released=False):
        super(MessageDispositionTask, self).__init__()
        self._disposition = disposition
        self._released = released

    def wait(self):
        pass

    def _execute(self, controller):
        try:
            self._disposition(self._released)
        except Exception as e:
            LOG.exception('Message acknowledgment failed: %s', e)