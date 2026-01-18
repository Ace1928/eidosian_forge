from __future__ import annotations
import time
import anyio
import inspect
import contextlib 
import functools
import hashlib
from lazyops.types.common import UpperStrEnum
from lazyops.utils import timed_cache
from lazyops.utils.helpers import create_background_task, fail_after
from lazyops.utils.lazy import lazy_import
from lazyops.utils.pooler import ThreadPooler
from lazyops.utils.lazy import get_function_name
from .compat import BaseModel, root_validator, get_pyd_dict
from .base import ENOVAL
from typing import Optional, Dict, Any, Callable, List, Union, TypeVar, Type, overload, TYPE_CHECKING
from aiokeydb.utils.logs import logger
from aiokeydb.utils.helpers import afail_after
@classmethod
def validate_decoder(cls, v) -> Optional[Callable]:
    """
        Returns the decoder
        """
    if v is None:
        from aiokeydb.serializers import DillSerializer
        return DillSerializer.loads
    v = cls.validate_callable(v)
    if not inspect.isfunction(v):
        if hasattr(v, 'loads') and inspect.isfunction(v.loads):
            return v.loads
        raise ValueError('Encoder must be callable or have a callable "dumps" method')
    return v