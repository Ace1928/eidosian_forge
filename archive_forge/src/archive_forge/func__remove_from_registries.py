import asyncio
import inspect
import json
import logging
import warnings
import zlib
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union
from uuid import uuid4
from redis import WatchError
from .defaults import CALLBACK_TIMEOUT, UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .timeouts import BaseDeathPenalty, JobTimeoutException
from .connections import resolve_connection
from .exceptions import DeserializationError, InvalidJobOperation, NoSuchJobError
from .local import LocalStack
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import (
def _remove_from_registries(self, pipeline: Optional['Pipeline']=None, remove_from_queue: bool=True):
    from .registry import BaseRegistry
    if remove_from_queue:
        from .queue import Queue
        q = Queue(name=self.origin, connection=self.connection, serializer=self.serializer)
        q.remove(self, pipeline=pipeline)
    registry: BaseRegistry
    if self.is_finished:
        from .registry import FinishedJobRegistry
        registry = FinishedJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
        registry.remove(self, pipeline=pipeline)
    elif self.is_deferred:
        from .registry import DeferredJobRegistry
        registry = DeferredJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
        registry.remove(self, pipeline=pipeline)
    elif self.is_started:
        from .registry import StartedJobRegistry
        registry = StartedJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
        registry.remove(self, pipeline=pipeline)
    elif self.is_scheduled:
        from .registry import ScheduledJobRegistry
        registry = ScheduledJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
        registry.remove(self, pipeline=pipeline)
    elif self.is_failed or self.is_stopped:
        self.failed_job_registry.remove(self, pipeline=pipeline)
    elif self.is_canceled:
        from .registry import CanceledJobRegistry
        registry = CanceledJobRegistry(self.origin, connection=self.connection, job_class=self.__class__, serializer=self.serializer)
        registry.remove(self, pipeline=pipeline)