import collections
import contextlib
import errno
import functools
import itertools
import math
import os
import random
import socket
import ssl
import sys
import threading
import time
from urllib import parse
import uuid
from amqp import exceptions as amqp_ex
import kombu
import kombu.connection
import kombu.entity
import kombu.messaging
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import eventletutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import amqpdriver
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging._drivers import pool
from oslo_messaging import _utils
from oslo_messaging import exceptions
def _heartbeat_thread_job(self):
    """Thread that maintains inactive connections
        """
    while not self._heartbeat_exit_event.is_set():
        with self._connection_lock.for_heartbeat():
            try:
                try:
                    self._heartbeat_check()
                    try:
                        self.connection.drain_events(timeout=0.001)
                    except socket.timeout:
                        pass
                except (socket.timeout, ConnectionRefusedError, OSError, kombu.exceptions.OperationalError, amqp_ex.ConnectionForced) as exc:
                    LOG.info('A recoverable connection/channel error occurred, trying to reconnect: %s', exc)
                    self.ensure_connection()
            except Exception:
                LOG.warning('Unexpected error during heartbeat thread processing, retrying...')
                LOG.debug('Exception', exc_info=True)
        self._heartbeat_exit_event.wait(timeout=self._heartbeat_wait_timeout)
    self._heartbeat_exit_event.clear()