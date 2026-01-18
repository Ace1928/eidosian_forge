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
def validate_kws(cls, values: Dict[str, Any], is_update: Optional[bool]=False) -> Dict[str, Any]:
    """
        Validates the attributes
        """
    if 'name' in values:
        values['name'] = cls.validate_callable(values.get('name'))
    if 'keybuilder' in values:
        values['keybuilder'] = cls.validate_callable(values.get('keybuilder'))
    if 'encoder' in values:
        values['encoder'] = cls.validate_encoder(values.get('encoder'))
    if 'decoder' in values:
        values['decoder'] = cls.validate_decoder(values.get('decoder'))
    if 'hit_setter' in values:
        values['hit_setter'] = cls.validate_callable(values.get('hit_setter'))
    if 'hit_getter' in values:
        values['hit_getter'] = cls.validate_callable(values.get('hit_getter'))
    if 'disabled' in values:
        values['disabled'] = cls.validate_callable(values.get('disabled'))
    if 'invalidate_if' in values:
        values['invalidate_if'] = cls.validate_callable(values.get('invalidate_if'))
    if 'invalidate_after' in values:
        values['invalidate_after'] = cls.validate_callable(values.get('invalidate_after'))
    if 'overwrite_if' in values:
        values['overwrite_if'] = cls.validate_callable(values.get('overwrite_if'))
    if 'bypass_if' in values:
        values['bypass_if'] = cls.validate_callable(values.get('bypass_if'))
    if 'post_init_hook' in values:
        values['post_init_hook'] = cls.validate_callable(values.get('post_init_hook'))
    if 'post_call_hook' in values:
        values['post_call_hook'] = cls.validate_callable(values.get('post_call_hook'))
    if 'cache_max_size' in values:
        values['cache_max_size'] = int(values['cache_max_size']) if values['cache_max_size'] else None
        if 'cache_max_size_policy' in values:
            values['cache_max_size_policy'] = CachePolicy(values['cache_max_size_policy'])
        elif not is_update:
            values['cache_max_size_policy'] = CachePolicy.LFU
    elif 'cache_max_size_policy' in values:
        values['cache_max_size_policy'] = CachePolicy(values['cache_max_size_policy'])
    return values