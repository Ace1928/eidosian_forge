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
class ConnectionLock(DummyConnectionLock):
    """Lock object to protect access to the kombu connection

    This is a lock object to protect access to the kombu connection
    object between the heartbeat thread and the driver thread.

    They are two way to acquire this lock:
        * lock.acquire()
        * lock.heartbeat_acquire()

    In both case lock.release(), release the lock.

    The goal is that the heartbeat thread always have the priority
    for acquiring the lock. This ensures we have no heartbeat
    starvation when the driver sends a lot of messages.

    So when lock.heartbeat_acquire() is called next time the lock
    is released(), the caller unconditionally acquires
    the lock, even someone else have asked for the lock before it.
    """

    def __init__(self):
        self._workers_waiting = 0
        self._heartbeat_waiting = False
        self._lock_acquired = None
        self._monitor = threading.Lock()
        self._workers_locks = threading.Condition(self._monitor)
        self._heartbeat_lock = threading.Condition(self._monitor)
        self._get_thread_id = eventletutils.fetch_current_thread_functor()

    def acquire(self):
        with self._monitor:
            while self._lock_acquired:
                self._workers_waiting += 1
                self._workers_locks.wait()
                self._workers_waiting -= 1
            self._lock_acquired = self._get_thread_id()

    def heartbeat_acquire(self):
        with self._monitor:
            while self._lock_acquired is not None:
                self._heartbeat_waiting = True
                self._heartbeat_lock.wait()
                self._heartbeat_waiting = False
            self._lock_acquired = self._get_thread_id()

    def release(self):
        with self._monitor:
            if self._lock_acquired is None:
                raise RuntimeError("We can't release a not acquired lock")
            thread_id = self._get_thread_id()
            if self._lock_acquired != thread_id:
                raise RuntimeError("We can't release lock acquired by another thread/greenthread; %s vs %s" % (self._lock_acquired, thread_id))
            self._lock_acquired = None
            if self._heartbeat_waiting:
                self._heartbeat_lock.notify()
            elif self._workers_waiting > 0:
                self._workers_locks.notify()

    @contextlib.contextmanager
    def for_heartbeat(self):
        self.heartbeat_acquire()
        try:
            yield
        finally:
            self.release()