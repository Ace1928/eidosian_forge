from __future__ import annotations
import typing
import logging
import datetime
from redis.commands import (
from redis.client import (
from redis.asyncio.client import (
from aiokeydb.v2.connection import (
from typing import Any, Iterable, Mapping, Callable, Union, overload, TYPE_CHECKING
from typing_extensions import Literal
from .utils.helpers import get_retryable_wrapper
class KeyDB(Redis):
    """
    Implementation of the KeyDB protocol.

    This abstract class provides a Python interface to all KeyDB commands
    and an implementation of the KeyDB protocol.

    Pipelines derive from this, implementing how
    the commands are sent and received to the KeyDB server. Based on
    configuration, an instance will either use a ConnectionPool, or
    Connection object to talk to keydb.

    It is not safe to pass PubSub or Pipeline objects between threads.
    """

    @property
    def is_async(self):
        return False

    @classmethod
    def from_url(cls, url, pool_class: typing.Optional[typing.Type[ConnectionPool]]=ConnectionPool, **kwargs):
        """
        Return a Redis client object configured from the given URL

        For example::
            keydb://[[username]:[password]]@localhost:6379/0
            keydbs://[[username]:[password]]@localhost:6379/0
            dfly://[[username]:[password]]@localhost:6379/0
            dflys://[[username]:[password]]@localhost:6379/0
            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[[username]:[password]]@/path/to/socket.sock?db=0

        Seven URL schemes are supported:
        - `keydb://` creates a TCP socket connection. (KeyDB)
        - `keydbs://` creates a SSL wrapped TCP socket connection. (KeyDB)
        - `dfly://` creates a TCP socket connection. (Dragonfly)
        - `dflys://` creates a SSL wrapped TCP socket connection. (Dragonfly)
        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.


        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

            1. A ``db`` querystring option, e.g. redis://localhost?db=0
            2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0
            3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.

        """
        if not pool_class:
            pool_class = ConnectionPool
        connection_pool = pool_class.from_url(url, **kwargs)
        return cls(connection_pool=connection_pool)

    def pubsub(self, retryable: typing.Optional[bool]=False, **kwargs) -> PubSubT:
        """
        Return a Publish/Subscribe object. With this object, you can
        subscribe to channels and listen for messages that get published to
        them.
        """
        pubsub_class = RetryablePubSub if retryable else PubSub
        return pubsub_class(connection_pool=self.connection_pool, **kwargs)

    def pipeline(self, transaction: bool=True, shard_hint: typing.Optional[str]=None, retryable: typing.Optional[bool]=False) -> PipelineT:
        """
        Return a new pipeline object that can queue multiple commands for
        later execution. ``transaction`` indicates whether all commands
        should be executed atomically. Apart from making a group of operations
        atomic, pipelines are useful for reducing the back-and-forth overhead
        between the client and server.
        """
        pipeline_class = RetryablePipeline if retryable else Pipeline
        return pipeline_class(connection_pool=self.connection_pool, response_callbacks=self.response_callbacks, transaction=transaction, shard_hint=shard_hint)