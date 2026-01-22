import collections
import logging
import os
import threading
import uuid
import warnings
from debtcollector import removals
from oslo_config import cfg
from oslo_messaging.target import Target
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_messaging._drivers.amqp1_driver.eventloop import compute_timeout
from oslo_messaging._drivers.amqp1_driver import opts
from oslo_messaging._drivers import base
from oslo_messaging._drivers import common
def _ensure_connect_called(func):
    """Causes a new controller to be created when the messaging service is
        first used by the current process. It is safe to push tasks to it
        whether connected or not, but those tasks won't be processed until
        connection completes.
        """

    def wrap(self, *args, **kws):
        with self._lock:
            old_pid = self._pid
            self._pid = os.getpid()
            if old_pid != self._pid:
                if self._ctrl is not None:
                    LOG.warning('Process forked after connection established!')
                    self._ctrl = None
                self._ctrl = controller.Controller(self._url, self._default_exchange, self._conf)
                self._ctrl.connect()
        return func(self, *args, **kws)
    return wrap