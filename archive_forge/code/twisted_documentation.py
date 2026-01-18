import socket
import sys
import twisted.internet.abstract  # type: ignore
import twisted.internet.asyncioreactor  # type: ignore
from twisted.internet.defer import Deferred  # type: ignore
from twisted.python import failure  # type: ignore
import twisted.names.cache  # type: ignore
import twisted.names.client  # type: ignore
import twisted.names.hosts  # type: ignore
import twisted.names.resolve  # type: ignore
from tornado.concurrent import Future, future_set_exc_info
from tornado.escape import utf8
from tornado import gen
from tornado.netutil import Resolver
import typing
Install ``AsyncioSelectorReactor`` as the default Twisted reactor.

    .. deprecated:: 5.1

       This function is provided for backwards compatibility; code
       that does not require compatibility with older versions of
       Tornado should use
       ``twisted.internet.asyncioreactor.install()`` directly.

    .. versionchanged:: 6.0.3

       In Tornado 5.x and before, this function installed a reactor
       based on the Tornado ``IOLoop``. When that reactor
       implementation was removed in Tornado 6.0.0, this function was
       removed as well. It was restored in Tornado 6.0.3 using the
       ``asyncio`` reactor instead.

    