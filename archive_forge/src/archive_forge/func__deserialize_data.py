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
def _deserialize_data(self):
    """Deserializes the Job `data` into a tuple.
        This includes the `_func_name`, `_instance`, `_args` and `_kwargs`

        Raises:
            DeserializationError: Cathes any deserialization error (since serializers are generic)
        """
    try:
        self._func_name, self._instance, self._args, self._kwargs = self.serializer.loads(self.data)
    except Exception as e:
        raise DeserializationError() from e