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
class PylibmcBackend(MemcacheArgs, GenericMemcachedBackend):
    """A backend for the
    `pylibmc <http://sendapatch.se/projects/pylibmc/index.html>`_
    memcached client.

    A configuration illustrating several of the optional
    arguments described in the pylibmc documentation::

        from dogpile.cache import make_region

        region = make_region().configure(
            'dogpile.cache.pylibmc',
            expiration_time = 3600,
            arguments = {
                'url':["127.0.0.1"],
                'binary':True,
                'behaviors':{"tcp_nodelay": True,"ketama":True}
            }
        )

    Arguments accepted here include those of
    :class:`.GenericMemcachedBackend`, as well as
    those below.

    :param binary: sets the ``binary`` flag understood by
     ``pylibmc.Client``.
    :param behaviors: a dictionary which will be passed to
     ``pylibmc.Client`` as the ``behaviors`` parameter.
    :param min_compress_len: Integer, will be passed as the
     ``min_compress_len`` parameter to the ``pylibmc.Client.set``
     method.

    """

    def __init__(self, arguments):
        self.binary = arguments.get('binary', False)
        self.behaviors = arguments.get('behaviors', {})
        super(PylibmcBackend, self).__init__(arguments)

    def _imports(self):
        global pylibmc
        import pylibmc

    def _create_client(self):
        return pylibmc.Client(self.url, binary=self.binary, behaviors=self.behaviors)