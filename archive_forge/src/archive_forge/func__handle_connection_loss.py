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
def _handle_connection_loss(self, reason):
    """The connection to the messaging service has been lost.  Try to
        reestablish the connection/failover if not shutting down the driver.
        """
    self.addresser = None
    self._socket_connection.close()
    if self._closing:
        self.processor.shutdown()
    else:
        if not self._reconnecting:
            self._reconnecting = True
            self.processor.wakeup(lambda: self._hard_reset(reason))
            LOG.info('Delaying reconnect attempt for %d seconds', self._delay)
            self.processor.defer(lambda: self._do_reconnect(reason), self._delay)
            self._delay = min(self._delay * self.conn_retry_backoff, self.conn_retry_interval_max)
        if self._link_maint_timer:
            self._link_maint_timer.cancel()
            self._link_maint_timer = None