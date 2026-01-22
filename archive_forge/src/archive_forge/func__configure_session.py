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
def _configure_session(cls, name: str='default', uri: str=None, host: str=None, port: int=None, db_id: int=None, username: str=None, password: str=None, protocol: str=None, with_auth: bool=True, cache_enabled: typing.Optional[bool]=None, encoder: typing.Optional[typing.Any]=None, serializer: typing.Optional[typing.Any]=None, loop: asyncio.AbstractEventLoop=None, **kwargs) -> KeyDBSession:
    """
        Configures a new session
        """
    uri: KeyDBUri = cls.settings.create_uri(name=name, uri=uri, host=host, port=port, db_id=db_id, username=username, password=password, protocol=protocol, with_auth=with_auth)
    db_id = db_id or uri.db_id
    db_id = db_id if db_id is not None else cls.settings.get_db_id(name=name, db=db_id)
    config = cls.settings.get_config(**kwargs)
    config['db'] = db_id
    serializer = None if serializer is False else serializer or cls.settings.get_serializer()
    pool = cls._configure_pool(name=name, uri=uri, serializer=serializer, loop=loop, **config)
    return KeyDBSession(uri=uri, name=name, client_pools=pool, serializer=serializer, encoder=None if encoder is False else encoder or cls.encoder, settings=cls.settings, cache_enabled=cache_enabled, **config)