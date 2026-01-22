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
class RabbitDriver(amqpdriver.AMQPDriverBase):
    """RabbitMQ Driver

    The ``rabbit`` driver is the default driver used in OpenStack's
    integration tests.

    The driver is aliased as ``kombu`` to support upgrading existing
    installations with older settings.

    """

    def __init__(self, conf, url, default_exchange=None, allowed_remote_exmods=None):
        opt_group = cfg.OptGroup(name='oslo_messaging_rabbit', title='RabbitMQ driver options')
        conf.register_group(opt_group)
        conf.register_opts(rabbit_opts, group=opt_group)
        conf.register_opts(rpc_amqp.amqp_opts, group=opt_group)
        conf.register_opts(base.base_opts, group=opt_group)
        conf = rpc_common.ConfigOptsProxy(conf, url, opt_group.name)
        self.missing_destination_retry_timeout = conf.oslo_messaging_rabbit.kombu_missing_consumer_retry_timeout
        self.prefetch_size = conf.oslo_messaging_rabbit.rabbit_qos_prefetch_count
        max_size = conf.oslo_messaging_rabbit.rpc_conn_pool_size
        min_size = conf.oslo_messaging_rabbit.conn_pool_min_size
        if max_size < min_size:
            raise RuntimeError(f'rpc_conn_pool_size: {max_size} must be greater than or equal to conn_pool_min_size: {min_size}')
        ttl = conf.oslo_messaging_rabbit.conn_pool_ttl
        connection_pool = pool.ConnectionPool(conf, max_size, min_size, ttl, url, Connection)
        super(RabbitDriver, self).__init__(conf, url, connection_pool, default_exchange, allowed_remote_exmods)

    def require_features(self, requeue=True):
        pass