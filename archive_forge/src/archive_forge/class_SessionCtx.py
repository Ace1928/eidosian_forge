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
class SessionCtx(BaseModel):
    """
    Holds the reference for session ctx
    """
    active: bool = False
    cache_max_attempts: int = 20
    cache_failed_attempts: int = 0
    client: typing.Optional[KeyDB] = None
    async_client: typing.Optional[AsyncKeyDB] = None
    lock: typing.Optional[Lock] = None
    async_lock: typing.Optional[AsyncLock] = None
    pipeline: typing.Optional[Pipeline] = None
    async_pipeline: typing.Optional[AsyncPipeline] = None
    locks: typing.Dict[str, Lock] = {}
    async_locks: typing.Dict[str, AsyncLock] = {}
    pubsub: typing.Optional[PubSub] = None
    async_pubsub: typing.Optional[AsyncPubSub] = None
    pipeline: typing.Optional[Pipeline] = None
    async_pipeline: typing.Optional[AsyncPipeline] = None

    class Config:
        arbitrary_types_allowed = True