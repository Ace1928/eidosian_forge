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
def wait_for_ready(self, interval: int=1.0, max_attempts: typing.Optional[int]=None, timeout: typing.Optional[float]=60.0, verbose: bool=False, **kwargs):
    if self.state.active:
        return
    attempts = 0
    start_time = time.time()
    while True:
        if max_attempts and attempts >= max_attempts:
            raise ConnectionError(f'[{self.name}] Max {max_attempts} attempts reached')
        if timeout and time.time() - start_time >= timeout:
            raise TimeoutError(f'[{self.name}] Timeout of {timeout} seconds reached')
        try:
            self.ping()
            if verbose:
                logger.info(f'[{self.name}] KeyDB is Ready after {attempts} attempts')
            self.state.active = True
            break
        except (InterruptedError, KeyboardInterrupt) as e:
            logger.error(e)
            break
        except Exception as e:
            if verbose:
                logger.info(f'[{self.name}] KeyDB is not ready, retrying in {interval} seconds')
            time.sleep(interval)
            attempts += 1