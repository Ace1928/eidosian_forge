import abc
import collections
import threading
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_messaging._drivers import common
def _on_expire(self, connection):
    connection.close()
    LOG.debug('Idle connection has expired and been closed. Pool size: %d' % len(self._items))