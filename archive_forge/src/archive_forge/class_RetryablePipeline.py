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
class RetryablePipeline(Pipeline):
    """
    Retryable Pipeline
    """

    @retryable_wrapper
    def execute(self, raise_on_error: bool=True) -> typing.List[typing.Any]:
        """Execute all the commands in the current pipeline"""
        return super().execute(raise_on_error=raise_on_error)