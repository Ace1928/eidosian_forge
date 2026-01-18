import sys
import time
import anyio
import typing
import logging
import asyncio
import functools
import contextlib
from pydantic import BaseModel
from pydantic.types import ByteSize
from aiokeydb.v2.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.v2.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.v2.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.v2.exceptions import (
from aiokeydb.v2.types import KeyDBUri, ENOVAL
from aiokeydb.v2.configs import KeyDBSettings, settings as default_settings
from aiokeydb.v2.utils import full_name, args_to_key
from aiokeydb.v2.utils.helpers import create_retryable_client
from aiokeydb.v2.serializers import BaseSerializer
from inspect import iscoroutinefunction
def with_db_id(self, db_id: int) -> 'KeyDBSession':
    """
        Initialize a new session with the given db_id
        if the db_id is different from the current one.
        """
    if db_id != self.db_id:
        return self.__class__(uri=self.uri, name=self.name, client_pools=self.client_pools.with_db_id(db_id), db_id=db_id, encoder=self.encoder, serializer=self.serializer, settings=self.settings, cache_ttl=self.cache_ttl, cache_prefix=self.cache_prefix, cache_enabled=self.cache_enabled, **self.config)
    return self