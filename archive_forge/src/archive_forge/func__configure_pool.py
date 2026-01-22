from __future__ import annotations
import asyncio
import typing
import logging
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.connection import Encoder, ConnectionPool, BlockingConnectionPool, Connection
from aiokeydb.v1.core import KeyDB, PubSub, Pipeline
from aiokeydb.v1.typing import Number, KeyT, AbsExpiryT, ExpiryT
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool, AsyncBlockingConnectionPool, AsyncConnection
from aiokeydb.v1.client.config import KeyDBSettings
from aiokeydb.v1.client.types import KeyDBUri
from aiokeydb.v1.client.schemas.session import KeyDBSession, ClientPools
from aiokeydb.v1.client.serializers import SerializerType, BaseSerializer
def _configure_pool(cls, name: str, uri: KeyDBUri, max_connections: int=None, pool_class: typing.Type[ConnectionPool]=BlockingConnectionPool, connection_class: typing.Type[Connection]=Connection, connection_kwargs: typing.Dict[str, typing.Any]=None, amax_connections: int=None, apool_class: typing.Type[AsyncConnectionPool]=AsyncBlockingConnectionPool, aconnection_class: typing.Type[AsyncConnection]=AsyncConnection, aconnection_kwargs: typing.Dict[str, typing.Any]=None, auto_pubsub: typing.Optional[bool]=True, pubsub_decode_responses: typing.Optional[bool]=True, decode_responses: typing.Optional[bool]=None, serializer: typing.Optional[typing.Any]=None, loop: asyncio.AbstractEventLoop=None, **config) -> ClientPools:
    """
        Configures the pool for the given session
        """
    if uri.key in cls.pools and loop is None:
        return cls.pools[uri.key]
    connection_kwargs = connection_kwargs or {}
    aconnection_kwargs = aconnection_kwargs or {}
    decode_responses = decode_responses if decode_responses is not None else not bool(serializer or cls.serializer)
    logger.log(msg=f'Configuring Pool for {name} w/ {uri.key}', level=cls.settings.loglevel)
    _pool = ClientPools(name=name, pool=pool_class.from_url(uri.connection, decode_responses=decode_responses, max_connections=max_connections, connection_class=connection_class, auto_pubsub=auto_pubsub, pubsub_decode_responses=pubsub_decode_responses, **connection_kwargs, **config), apool=apool_class.from_url(uri.connection, decode_responses=decode_responses, max_connections=amax_connections, connection_class=aconnection_class, auto_pubsub=auto_pubsub, pubsub_decode_responses=pubsub_decode_responses, **aconnection_kwargs, **config))
    cls.pools[uri.key] = _pool
    return _pool