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
def _process_tasks(self):
    """Execute Task objects in the context of the processor thread."""
    with self._process_tasks_lock:
        self._process_tasks_scheduled = False
    count = 0
    while not self._tasks.empty() and count < self._max_task_batch:
        try:
            self._tasks.get(False)._execute(self)
        except Exception as e:
            LOG.exception('Error processing task: %s', e)
        count += 1
    if not self._tasks.empty():
        self._schedule_task_processing()