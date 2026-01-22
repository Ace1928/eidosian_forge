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
class BMemcachedBackend(GenericMemcachedBackend):
    """A backend for the
    `python-binary-memcached <https://github.com/jaysonsantos/    python-binary-memcached>`_
    memcached client.

    This is a pure Python memcached client which includes
    security features like SASL and SSL/TLS.

    SASL is a standard for adding authentication mechanisms
    to protocols in a way that is protocol independent.

    A typical configuration using username/password::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.bmemcached',
            expiration_time = 3600,
            arguments = {
                'url':["127.0.0.1"],
                'username':'scott',
                'password':'tiger'
            }
        )

    A typical configuration using tls_context::

        import ssl
        from dogpile.cache import make_region

        ctx = ssl.create_default_context(cafile="/path/to/my-ca.pem")

        region = make_region().configure(
            'dogpile.cache.bmemcached',
            expiration_time = 3600,
            arguments = {
                'url':["127.0.0.1"],
                'tls_context':ctx,
            }
        )

    For advanced ways to configure TLS creating a more complex
    tls_context visit https://docs.python.org/3/library/ssl.html

    Arguments which can be passed to the ``arguments``
    dictionary include:

    :param username: optional username, will be used for
     SASL authentication.
    :param password: optional password, will be used for
     SASL authentication.
    :param tls_context: optional TLS context, will be used for
     TLS connections.

     .. versionadded:: 1.0.2

    """

    def __init__(self, arguments):
        self.username = arguments.get('username', None)
        self.password = arguments.get('password', None)
        self.tls_context = arguments.get('tls_context', None)
        super(BMemcachedBackend, self).__init__(arguments)

    def _imports(self):
        global bmemcached
        import bmemcached

        class RepairBMemcachedAPI(bmemcached.Client):
            """Repairs BMemcached's non-standard method
            signatures, which was fixed in BMemcached
            ef206ed4473fec3b639e.

            """

            def add(self, key, value, timeout=0):
                try:
                    return super(RepairBMemcachedAPI, self).add(key, value, timeout)
                except ValueError:
                    return False
        self.Client = RepairBMemcachedAPI

    def _create_client(self):
        return self.Client(self.url, username=self.username, password=self.password, tls_context=self.tls_context)

    def delete_multi(self, keys):
        """python-binary-memcached api does not implements delete_multi"""
        for key in keys:
            self.delete(key)