import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class MemcachedBackend(MemcacheArgs, GenericMemcachedBackend):
    """A backend using the standard
    `Python-memcached <http://www.tummy.com/Community/software/    python-memcached/>`_
    library.

    Example::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.memcached',
            expiration_time = 3600,
            arguments = {
                'url':"127.0.0.1:11211"
            }
        )

    :param dead_retry: Number of seconds memcached server is considered dead
     before it is tried again. Will be passed to ``memcache.Client``
     as the ``dead_retry`` parameter.

     .. versionchanged:: 1.1.8  Moved the ``dead_retry`` argument which was
        erroneously added to "set_parameters" to
        be part of the Memcached connection arguments.

    :param socket_timeout: Timeout in seconds for every call to a server.
      Will be passed to ``memcache.Client`` as the ``socket_timeout``
      parameter.

      .. versionchanged:: 1.1.8  Moved the ``socket_timeout`` argument which
         was erroneously added to "set_parameters"
         to be part of the Memcached connection arguments.

    """

    def __init__(self, arguments):
        self.dead_retry = arguments.get('dead_retry', 30)
        self.socket_timeout = arguments.get('socket_timeout', 3)
        super(MemcachedBackend, self).__init__(arguments)

    def _imports(self):
        global memcache
        import memcache

    def _create_client(self):
        return memcache.Client(self.url, dead_retry=self.dead_retry, socket_timeout=self.socket_timeout)