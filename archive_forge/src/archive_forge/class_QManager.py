import logging
import os
import queue
import threading
import time
import uuid
import cachetools
from oslo_concurrency import lockutils
from oslo_utils import eventletutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging._drivers import amqp as rpc_amqp
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common as rpc_common
from oslo_messaging import MessageDeliveryFailure
class QManager(object):
    """Queue Manager to build queue name for reply (and fanout) type.
    This class is used only when use_queue_manager is set to True in config
    file.
    It rely on a shared memory accross processes (reading/writing data to
    /dev/shm/xyz) and oslo_concurrency.lockutils to avoid assigning the same
    queue name twice (or more) to different processes.
    The original idea of this queue manager was to avoid random queue names,
    so on service restart, the previously created queues can be reused,
    avoiding deletion/creation of queues on rabbitmq side (which cost a lot
    at scale).
    """

    def __init__(self, hostname, processname):
        self.hostname = hostname
        self.processname = processname
        self.file_name = '/dev/shm/%s_%s_qmanager' % (self.hostname, self.processname)
        self.pg = os.getpgrp()

    def get(self):
        lock_name = 'oslo_read_shm_%s_%s' % (self.hostname, self.processname)

        @lockutils.synchronized(lock_name, external=True)
        def read_from_shm():
            try:
                with open(self.file_name, 'r') as f:
                    pg, c = f.readline().split(':')
                    pg = int(pg)
                    c = int(c)
            except (FileNotFoundError, ValueError):
                pg = self.pg
                c = 0
            if pg == self.pg:
                c += 1
            else:
                c = 1
            with open(self.file_name, 'w') as f:
                f.write(str(self.pg) + ':' + str(c))
            return c
        c = read_from_shm()
        return self.hostname + ':' + self.processname + ':' + str(c)