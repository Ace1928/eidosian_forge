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
@classmethod
def _get_client_class(cls, settings: KeyDBSettings, retry_client_enabled: typing.Optional[bool]=None, retry_client_max_attempts: typing.Optional[int]=None, retry_client_max_delay: typing.Optional[int]=None, retry_client_logging_level: typing.Optional[int]=None, **kwargs) -> typing.Type[KeyDB]:
    """
        Returns the client class
        """
    retry_client_enabled = retry_client_enabled if retry_client_enabled is not None else settings.retry_client_enabled
    if not retry_client_enabled:
        return KeyDB
    retry_client_max_attempts = retry_client_max_attempts if retry_client_max_attempts is not None else settings.retry_client_max_attempts
    retry_client_max_delay = retry_client_max_delay if retry_client_max_delay is not None else settings.retry_client_max_delay
    retry_client_logging_level = retry_client_logging_level if retry_client_logging_level is not None else settings.retry_client_logging_level
    return create_retryable_client(KeyDB, max_attempts=retry_client_max_attempts, max_delay=retry_client_max_delay, logging_level=retry_client_logging_level, **kwargs)