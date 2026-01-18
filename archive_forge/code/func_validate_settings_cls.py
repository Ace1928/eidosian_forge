import json
import functools
from lazyops.types.models import BaseModel, validator
from lazyops.types.classprops import lazyproperty
from lazyops.types.static import RESPONSE_SUCCESS_CODES
from lazyops.types.resources import BaseResource, ResourceType, ResponseResource, ResponseResourceType
from lazyops.types.errors import ClientError, fatal_exception
from lazyops.imports._aiohttpx import aiohttpx, resolve_aiohttpx
from lazyops.imports._backoff import backoff, require_backoff
from lazyops.configs.base import DefaultSettings
from lazyops.utils.logs import default_logger as logger
from lazyops.utils.serialization import ObjectEncoder
from typing import Optional, Dict, List, Any, Type, Callable
@validator('settings_cls', pre=True)
def validate_settings_cls(cls, v):
    if v is None:
        v = DefaultSettings
    return v