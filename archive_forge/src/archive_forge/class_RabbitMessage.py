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
class RabbitMessage(dict):

    def __init__(self, raw_message):
        super(RabbitMessage, self).__init__(rpc_common.deserialize_msg(raw_message.payload))
        LOG.trace('RabbitMessage.Init: message %s', self)
        self._raw_message = raw_message

    def acknowledge(self):
        LOG.trace('RabbitMessage.acknowledge: message %s', self)
        self._raw_message.ack()

    def requeue(self):
        LOG.trace('RabbitMessage.requeue: message %s', self)
        self._raw_message.requeue()