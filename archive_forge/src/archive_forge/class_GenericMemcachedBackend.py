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
class GenericMemcachedBackend(CacheBackend):
    """Base class for memcached backends.

    This base class accepts a number of paramters
    common to all backends.

    :param url: the string URL to connect to.  Can be a single
     string or a list of strings.  This is the only argument
     that's required.
    :param distributed_lock: boolean, when True, will use a
     memcached-lock as the dogpile lock (see :class:`.MemcachedLock`).
     Use this when multiple
     processes will be talking to the same memcached instance.
     When left at False, dogpile will coordinate on a regular
     threading mutex.
    :param lock_timeout: integer, number of seconds after acquiring a lock that
     memcached should expire it.  This argument is only valid when
     ``distributed_lock`` is ``True``.

     .. versionadded:: 0.5.7

    :param memcached_expire_time: integer, when present will
     be passed as the ``time`` parameter to ``pylibmc.Client.set``.
     This is used to set the memcached expiry time for a value.

     .. note::

         This parameter is **different** from Dogpile's own
         ``expiration_time``, which is the number of seconds after
         which Dogpile will consider the value to be expired.
         When Dogpile considers a value to be expired,
         it **continues to use the value** until generation
         of a new value is complete, when using
         :meth:`.CacheRegion.get_or_create`.
         Therefore, if you are setting ``memcached_expire_time``, you'll
         want to make sure it is greater than ``expiration_time``
         by at least enough seconds for new values to be generated,
         else the value won't be available during a regeneration,
         forcing all threads to wait for a regeneration each time
         a value expires.

    The :class:`.GenericMemachedBackend` uses a ``threading.local()``
    object to store individual client objects per thread,
    as most modern memcached clients do not appear to be inherently
    threadsafe.

    In particular, ``threading.local()`` has the advantage over pylibmc's
    built-in thread pool in that it automatically discards objects
    associated with a particular thread when that thread ends.

    """
    set_arguments: Mapping[str, Any] = {}
    'Additional arguments which will be passed\n    to the :meth:`set` method.'
    serializer = None
    deserializer = None

    def __init__(self, arguments):
        self._imports()
        self.url = util.to_list(arguments['url'])
        self.distributed_lock = arguments.get('distributed_lock', False)
        self.lock_timeout = arguments.get('lock_timeout', 0)
        self.memcached_expire_time = arguments.get('memcached_expire_time', 0)

    def has_lock_timeout(self):
        return self.lock_timeout != 0

    def _imports(self):
        """client library imports go here."""
        raise NotImplementedError()

    def _create_client(self):
        """Creation of a Client instance goes here."""
        raise NotImplementedError()

    @util.memoized_property
    def _clients(self):
        backend = self

        class ClientPool(threading.local):

            def __init__(self):
                self.memcached = backend._create_client()
        return ClientPool()

    @property
    def client(self):
        """Return the memcached client.

        This uses a threading.local by
        default as it appears most modern
        memcached libs aren't inherently
        threadsafe.

        """
        return self._clients.memcached

    def get_mutex(self, key):
        if self.distributed_lock:
            return MemcachedLock(lambda: self.client, key, timeout=self.lock_timeout)
        else:
            return None

    def get(self, key):
        value = self.client.get(key)
        if value is None:
            return NO_VALUE
        else:
            return value

    def get_multi(self, keys):
        values = self.client.get_multi(keys)
        return [NO_VALUE if val is None else val for val in [values.get(key, NO_VALUE) for key in keys]]

    def set(self, key, value):
        self.client.set(key, value, **self.set_arguments)

    def set_multi(self, mapping):
        mapping = {key: value for key, value in mapping.items()}
        self.client.set_multi(mapping, **self.set_arguments)

    def delete(self, key):
        self.client.delete(key)

    def delete_multi(self, keys):
        self.client.delete_multi(keys)