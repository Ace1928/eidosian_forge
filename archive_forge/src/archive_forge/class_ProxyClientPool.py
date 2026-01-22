from __future__ import (absolute_import, division, print_function)
import collections
import os
import time
from multiprocessing import Lock
from itertools import chain
from ansible.errors import AnsibleError
from ansible.module_utils.common._collections_compat import MutableSet
from ansible.plugins.cache import BaseCacheModule
from ansible.utils.display import Display
class ProxyClientPool(object):
    """
    Memcached connection pooling for thread/fork safety. Inspired by py-redis
    connection pool.

    Available connections are maintained in a deque and released in a FIFO manner.
    """

    def __init__(self, *args, **kwargs):
        self.max_connections = kwargs.pop('max_connections', 1024)
        self.connection_args = args
        self.connection_kwargs = kwargs
        self.reset()

    def reset(self):
        self.pid = os.getpid()
        self._num_connections = 0
        self._available_connections = collections.deque(maxlen=self.max_connections)
        self._locked_connections = set()
        self._lock = Lock()

    def _check_safe(self):
        if self.pid != os.getpid():
            with self._lock:
                if self.pid == os.getpid():
                    return
                self.disconnect_all()
                self.reset()

    def get_connection(self):
        self._check_safe()
        try:
            connection = self._available_connections.popleft()
        except IndexError:
            connection = self.create_connection()
        self._locked_connections.add(connection)
        return connection

    def create_connection(self):
        if self._num_connections >= self.max_connections:
            raise RuntimeError('Too many memcached connections')
        self._num_connections += 1
        return memcache.Client(*self.connection_args, **self.connection_kwargs)

    def release_connection(self, connection):
        self._check_safe()
        self._locked_connections.remove(connection)
        self._available_connections.append(connection)

    def disconnect_all(self):
        for conn in chain(self._available_connections, self._locked_connections):
            conn.disconnect_all()

    def __getattr__(self, name):

        def wrapped(*args, **kwargs):
            return self._proxy_client(name, *args, **kwargs)
        return wrapped

    def _proxy_client(self, name, *args, **kwargs):
        conn = self.get_connection()
        try:
            return getattr(conn, name)(*args, **kwargs)
        finally:
            self.release_connection(conn)