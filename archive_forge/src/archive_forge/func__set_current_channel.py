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
def _set_current_channel(self, new_channel):
    """Change the channel to use.

        NOTE(sileht): Must be called within the connection lock
        """
    if new_channel == self.channel:
        return
    if self.channel is not None:
        self._declared_queues.clear()
        self._declared_exchanges.clear()
        self.connection.maybe_close_channel(self.channel)
    self.channel = new_channel
    if new_channel is not None:
        if self.purpose == rpc_common.PURPOSE_LISTEN:
            self._set_qos(new_channel)
        self._producer = kombu.messaging.Producer(new_channel, on_return=self.on_return)
        for consumer in self._consumers:
            consumer.declare(self)