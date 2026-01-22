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
class CachifyKwargs(BaseModel):
    """
    Cachify Config
    """
    ttl: Optional[int] = 60 * 10
    keybuilder: Optional[Callable] = None
    name: Optional[Union[str, Callable]] = None
    typed: Optional[bool] = True
    exclude_keys: Optional[List[str]] = None
    exclude_null: Optional[bool] = True
    exclude_exceptions: Optional[Union[bool, List[Exception]]] = True
    exclude_null_values_in_hash: Optional[bool] = None
    exclude_default_values_in_hash: Optional[bool] = None
    disabled: Optional[Union[bool, Callable]] = None
    invalidate_after: Optional[Union[int, Callable]] = None
    invalidate_if: Optional[Callable] = None
    overwrite_if: Optional[Callable] = None
    bypass_if: Optional[Callable] = None
    timeout: Optional[float] = 5.0
    verbose: Optional[bool] = False
    super_verbose: Optional[bool] = False
    raise_exceptions: Optional[bool] = True
    encoder: Optional[Union[str, Callable]] = None
    decoder: Optional[Union[str, Callable]] = None
    hit_setter: Optional[Callable] = None
    hit_getter: Optional[Callable] = None
    cache_max_size: Optional[int] = None
    cache_max_size_policy: Optional[Union[str, CachePolicy]] = CachePolicy.LFU
    post_init_hook: Optional[Union[str, Callable]] = None
    post_call_hook: Optional[Union[str, Callable]] = None
    cache_field: Optional[str] = None
    is_class_method: Optional[bool] = None
    has_ran_post_init_hook: Optional[bool] = None
    is_async: Optional[bool] = None
    hset_enabled: Optional[bool] = True
    if TYPE_CHECKING:
        session: Optional['KeyDBSession'] = None
    else:
        session: Optional[Any] = None

    @classmethod
    def validate_callable(cls, v: Optional[Union[str, int, Callable]]) -> Optional[Union[Callable, Any]]:
        """
        Validates the callable
        """
        return lazy_import(v) if isinstance(v, str) else v

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

    @classmethod
    def validate_encoder(cls, v) -> Optional[Callable]:
        """
        Returns the encoder
        """
        if v is None:
            from aiokeydb.serializers import DillSerializer
            return DillSerializer.dumps
        v = cls.validate_callable(v)
        if not inspect.isfunction(v):
            if hasattr(v, 'dumps') and inspect.isfunction(v.dumps):
                return v.dumps
            raise ValueError('Encoder must be callable or have a callable "dumps" method')
        return v

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

    class Config:
        """
        Config for CachifyKwargs
        """
        extra = 'ignore'
        arbitrary_types_allowed = True

    @root_validator(mode='after')
    def validate_attrs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the attributes
        """
        return cls.validate_kws(values)

    def update(self, **kwargs):
        """
        Validates and updates the kwargs
        """
        kwargs = self.validate_kws(kwargs, is_update=True)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                continue
            setattr(self, k, v)

    def get_key(self, key: str) -> str:
        """
        Gets the Key
        """
        return key if self.hset_enabled else f'{self.cache_field}:{key}'

    def build_hash_name(self, func: Callable, *args, **kwargs) -> str:
        """
        Builds the name for the function
        """
        if self.cache_field is not None:
            return self.cache_field
        if self.name:
            self.cache_field = self.name(func, *args, **kwargs) if callable(self.name) else self.name
        else:
            func = inspect.unwrap(func)
            self.cache_field = f'{func.__module__}.{func.__qualname__}'
        return self.cache_field

    async def abuild_hash_name(self, func: Callable, *args, **kwargs) -> str:
        """
        Builds the name for the function
        """
        if self.cache_field is not None:
            return self.cache_field
        if self.name:
            self.cache_field = await run_as_coro(self.name, func, *args, **kwargs) if callable(self.name) else self.name
        else:
            func = inspect.unwrap(func)
            self.cache_field = f'{func.__module__}.{func.__qualname__}'
        return self.cache_field

    def build_hash_key(self, *args, **kwargs) -> str:
        """
        Builds the key for the function
        """
        hash_func = self.keybuilder or hash_key
        return hash_func(args=args, kwargs=kwargs, typed=self.typed, exclude_keys=self.exclude_keys, exclude_null_values=self.exclude_null_values_in_hash, exclude_default_values=self.exclude_default_values_in_hash, is_class_method=self.is_class_method)

    async def abuild_hash_key(self, *args, **kwargs) -> str:
        """
        Builds the key for the function
        """
        hash_func = self.keybuilder or hash_key
        return await run_as_coro(hash_func, args=args, kwargs=kwargs, typed=self.typed, exclude_keys=self.exclude_keys, exclude_null_values=self.exclude_null_values_in_hash, exclude_default_values=self.exclude_default_values_in_hash, is_class_method=self.is_class_method)

    def should_cache(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the function should be cached
        """
        if self.disabled is not None:
            return not self.disabled
        return not self.disabled(*args, **kwargs) if callable(self.disabled) else True

    async def ashould_cache(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the function should be cached
        """
        if self.disabled is not None:
            return not self.disabled
        return not await run_as_coro(self.disabled, *args, **kwargs) if callable(self.disabled) else True

    def should_cache_value(self, val: Any) -> bool:
        """
        Returns whether or not the value should be cached
        """
        if self.exclude_null and val is None:
            return False
        if self.exclude_exceptions:
            if isinstance(self.exclude_exceptions, list):
                return not isinstance(val, tuple(self.exclude_exceptions))
            if isinstance(val, Exception):
                return False
        return True

    async def ashould_cache_value(self, val: Any) -> bool:
        """
        Returns whether or not the value should be cached
        """
        if self.exclude_null and val is None:
            return False
        if self.exclude_exceptions:
            if isinstance(self.exclude_exceptions, list):
                return not isinstance(val, tuple(self.exclude_exceptions))
            if isinstance(val, Exception):
                return False
        return True

    def should_invalidate(self, *args, _hits: Optional[int]=None, **kwargs) -> bool:
        """
        Returns whether or not the function should be invalidated
        """
        if self.invalidate_if is not None:
            return self.invalidate_if(*args, **kwargs)
        if self.invalidate_after is not None:
            if _hits and isinstance(self.invalidate_after, int):
                return _hits >= self.invalidate_after
            return self.invalidate_after(*args, _hits=_hits, **kwargs)
        return False

    async def ashould_invalidate(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the function should be invalidated
        """
        if self.invalidate_if is not None:
            return await run_as_coro(self.invalidate_if, *args, **kwargs)
        if self.invalidate_after is not None:
            _hits = await self.anum_hits
            if _hits and isinstance(self.invalidate_after, int):
                return _hits >= self.invalidate_after
            return await run_as_coro(self.invalidate_after, *args, _hits=_hits, **kwargs)
        return False

    def should_bypass(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the cache should be bypassed, returning 
        a fresh value from the function call
        """
        return self.bypass_if(*args, **kwargs) if self.bypass_if is not None else False

    async def ashould_bypass(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the cache should be bypassed, returning 
        a fresh value from the function call
        """
        if self.bypass_if is not None:
            return await run_as_coro(self.bypass_if, *args, **kwargs)
        return False
    '\n    v3 Methods\n    '

    def _get(self, key: str) -> Any:
        """
        Fetches the value from the cache
        """
        if self.hset_enabled:
            return self.session.client.hget(self.cache_field, key)
        return self.session.client.get(self.get_key(key))

    async def _aget(self, key: str) -> Any:
        """
        Fetches the value from the cache
        """
        if self.hset_enabled:
            return await self.session.async_client.hget(self.cache_field, key)
        return await self.session.async_client.get(self.get_key(key))

    def _set(self, key: str, value: Any) -> None:
        """
        Sets the value in the cache
        """
        if self.hset_enabled:
            return self.session.client.hset(self.cache_field, key, value)
        return self.session.client.set(self.get_key(key), value)

    async def _aset(self, key: str, value: Any) -> None:
        """
        Sets the value in the cache
        """
        if self.hset_enabled:
            return await self.session.async_client.hset(self.cache_field, key, value)
        return await self.session.async_client.set(self.get_key(key), value)

    def _delete(self, key: str) -> None:
        """
        Deletes the value in the cache
        """
        if self.hset_enabled:
            return self.session.client.hdel(self.cache_field, key)
        return self.session.client.delete(self.get_key(key))

    async def _adelete(self, key: str) -> None:
        """
        Deletes the value in the cache
        """
        if self.hset_enabled:
            return await self.session.async_client.hdel(self.cache_field, key)
        return await self.session.async_client.delete(self.get_key(key))

    def _clear(self, *keys: str) -> None:
        """
        Clears the keys in the cache
        """
        if self.hset_enabled:
            if keys:
                return self.session.client.hdel(self.cache_field, *keys)
            return self.session.client.delete(self.cache_field)
        if keys:
            return self.session.client.delete(*[self.get_key(k) for k in keys])
        return self.session.client.delete(self.get_key(self.cache_field, '*'))

    async def _aclear(self, *keys: str) -> None:
        """
        Clears the keys in the cache
        """
        if self.hset_enabled:
            if keys:
                return await self.session.async_client.hdel(self.cache_field, *keys)
            return await self.session.async_client.delete(self.cache_field)
        if keys:
            return await self.session.async_client.delete(*[self.get_key(k) for k in keys])
        return await self.session.async_client.delete(self.get_key(self.cache_field, '*'))

    def _exists(self, key: str) -> bool:
        """
        Returns whether or not the key exists
        """
        if self.hset_enabled:
            return self.session.client.hexists(self.cache_field, key)
        return self.session.client.exists(self.get_key(key))

    async def _aexists(self, key: str) -> bool:
        """
        Returns whether or not the key exists
        """
        if self.hset_enabled:
            return await self.session.async_client.hexists(self.cache_field, key)
        return await self.session.async_client.exists(self.get_key(key))

    def _expire(self, key: str, ttl: int) -> None:
        """
        Expires the key
        """
        if self.hset_enabled:
            return self.session.client.expire(self.cache_field, ttl)
        return self.session.client.expire(self.get_key(key), ttl)

    async def _aexpire(self, key: str, ttl: int) -> None:
        """
        Expires the key
        """
        if self.hset_enabled:
            return await self.session.async_client.expire(self.cache_field, ttl)
        return await self.session.async_client.expire(self.get_key(key), ttl)

    def _length(self) -> int:
        """
        Returns the size of the cache
        """
        if self.hset_enabled:
            return self.session.client.hlen(self.cache_field)
        return len(self.session.client.keys(self.get_key(self.cache_field, '*')))

    async def _alength(self) -> int:
        """
        Returns the size of the cache
        """
        if self.hset_enabled:
            return await self.session.async_client.hlen(self.cache_field)
        return len(await self.session.async_client.keys(self.get_key(self.cache_field, '*')))

    def _keys(self, decode: Optional[bool]=True) -> List[str]:
        """
        Returns the keys
        """
        if self.hset_enabled:
            keys = self.session.client.hkeys(self.cache_field)
        else:
            keys = self.session.client.keys(self.get_key(self.cache_field, '*'))
        if keys and decode:
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        return keys or []

    async def _akeys(self, decode: Optional[bool]=True) -> List[str]:
        """
        Returns the keys
        """
        if self.hset_enabled:
            keys = await self.session.async_client.hkeys(self.cache_field)
        else:
            keys = await self.session.async_client.keys(self.get_key(self.cache_field, '*'))
        if keys and decode:
            return [k.decode() if isinstance(k, bytes) else k for k in keys]
        return keys or []

    def _values(self, decode: Optional[bool]=False) -> List[Any]:
        """
        Returns the values
        """
        if self.hset_enabled:
            values = self.session.client.hvals(self.cache_field)
        else:
            values = self.session.client.mget(self._keys(decode=False))
        if values and decode:
            return [v.decode() if isinstance(v, bytes) else v for v in values]
        return values or []

    async def _avalues(self, decode: Optional[bool]=False) -> List[Any]:
        """
        Returns the values
        """
        if self.hset_enabled:
            values = await self.session.async_client.hvals(self.cache_field)
        else:
            values = await self.session.async_client.mget(self._keys(decode=False))
        if values and decode:
            return [v.decode() if isinstance(v, bytes) else v for v in values]
        return values or []

    def _items(self, decode: Optional[bool]=True) -> Dict[str, Any]:
        """
        Returns the items
        """
        if self.hset_enabled:
            items = self.session.client.hgetall(self.cache_field)
        else:
            items = self.session.client.mget(self._keys(decode=False))
        if items and decode:
            return {k.decode() if isinstance(k, bytes) else k: self.decode(v) for k, v in items.items()}
        return items or {}

    async def _aitems(self, decode: Optional[bool]=True) -> Dict[str, Any]:
        """
        Returns the items
        """
        if self.hset_enabled:
            items = await self.session.async_client.hgetall(self.cache_field)
        else:
            items = await self.session.async_client.mget(self._keys(decode=False))
        if items and decode:
            return {k.decode() if isinstance(k, bytes) else k: self.decode(v) for k, v in items.items()}
        return items or {}

    def _incr(self, key: str, amount: int=1) -> int:
        """
        Increments the key
        """
        if self.hset_enabled:
            return self.session.client.hincrby(self.cache_field, key, amount)
        return self.session.client.incr(self.get_key(key), amount)

    async def _aincr(self, key: str, amount: int=1) -> int:
        """
        Increments the key
        """
        if self.hset_enabled:
            return await self.session.async_client.hincrby(self.cache_field, key, amount)
        return await self.session.async_client.incr(self.get_key(key), amount)
    '\n    Props\n    '

    @property
    def has_post_init_hook(self) -> bool:
        """
        Returns whether or not there is a post init hook
        """
        return self.post_init_hook is not None

    @property
    def has_post_call_hook(self) -> bool:
        """
        Returns whether or not there is a post call hook
        """
        return self.post_call_hook is not None

    @property
    def num_default_keys(self) -> int:
        """
        Returns the number of default keys
        """
        n = 1
        if self.cache_max_size is not None:
            n += 2
        return n

    @property
    async def anum_hits(self) -> int:
        """
        Returns the number of hits
        """
        async with asafely(timeout=self.timeout):
            val = await self._aget('hits')
            return int(val) if val else 0

    @property
    async def anum_keys(self) -> int:
        """
        Returns the number of keys
        """
        async with asafely(timeout=self.timeout):
            val = await self._alength()
            return max(int(val) - self.num_default_keys, 0) if val else 0

    @property
    async def acache_keys(self) -> List[str]:
        """
        Returns the keys
        """
        async with asafely(timeout=self.timeout):
            return await self._akeys()

    @property
    async def acache_values(self) -> List[Any]:
        """
        Returns the values
        """
        async with asafely(timeout=self.timeout):
            return await self._avalues()

    @property
    async def acache_items(self) -> Dict[str, Any]:
        """
        Returns the items
        """
        async with asafely(timeout=self.timeout):
            return await self._aitems()

    @property
    async def acache_keyhits(self) -> Dict[str, int]:
        """
        Returns the size of the cache
        """
        async with asafely(timeout=self.timeout):
            val = await self._aget('keyhits')
            return {k.decode(): int(v) for k, v in val.items()} if val else {}

    @property
    async def acache_timestamps(self) -> Dict[str, float]:
        """
        Returns the size of the cache
        """
        async with asafely(timeout=self.timeout):
            val = await self._aget('timestamps')
            return {k.decode(): float(v) for k, v in val.items()} if val else {}

    @property
    async def acache_info(self) -> Dict[str, Any]:
        """
        Returns the info for the cache
        """
        return {'name': self.cache_field, 'hits': await self.anum_hits, 'keys': await self.anum_keys, 'keyhits': await self.acache_keyhits, 'timestamps': await self.acache_timestamps, 'max_size': self.cache_max_size, 'max_size_policy': self.cache_max_size_policy}

    @property
    def num_hits(self) -> int:
        """
        Returns the number of hits
        """
        with safely(timeout=self.timeout):
            val = self._get('hits')
            return int(val) if val else 0

    @property
    def num_keys(self) -> int:
        """
        Returns the number of keys
        """
        with safely(timeout=self.timeout):
            val = self._length()
            return max(int(val) - self.num_default_keys, 0) if val else 0

    @property
    def cache_keys(self) -> List[str]:
        """
        Returns the keys
        """
        with safely(timeout=self.timeout):
            return self._keys()

    @property
    def cache_values(self) -> List[Any]:
        """
        Returns the values
        """
        with safely(timeout=self.timeout):
            return self._values()

    @property
    def cache_items(self) -> Dict[str, Any]:
        """
        Returns the items
        """
        with safely(timeout=self.timeout):
            return self._items()

    @property
    def cache_keyhits(self) -> Dict[str, int]:
        """
        Returns the size of the cache
        """
        with safely(timeout=self.timeout):
            val = self._get('keyhits')
            return {k.decode(): int(v) for k, v in val.items()} if val else {}

    @property
    def cache_timestamps(self) -> Dict[str, float]:
        """
        Returns the size of the cache
        """
        with safely(timeout=self.timeout):
            val = self._get('timestamps')
            return {k.decode(): float(v) for k, v in val.items()} if val else {}

    @property
    def cache_info(self) -> Dict[str, Any]:
        """
        Returns the info for the cache
        """
        return {'name': self.cache_field, 'hits': self.num_hits, 'keys': self.num_keys, 'keyhits': self.cache_keyhits, 'timestamps': self.cache_timestamps, 'max_size': self.cache_max_size, 'max_size_policy': self.cache_max_size_policy}
    '\n    Methods\n    '

    def encode(self, value: Any) -> bytes:
        """
        Encodes the value
        """
        return self.encoder(value)

    def decode(self, value: bytes) -> Any:
        """
        Decodes the value
        """
        return self.decoder(value)

    def invalidate_cache(self, key: str) -> int:
        """
        Invalidates the cache
        """
        with safely(timeout=self.timeout):
            return self._delete(key, 'hits', 'timestamps', 'keyhits')

    async def ainvalidate_cache(self, key: str) -> int:
        """
        Invalidates the cache
        """
        async with asafely(timeout=self.timeout):
            return await self._adelete(key, 'hits', 'timestamps', 'keyhits')

    async def aadd_key_hit(self, key: str):
        """
        Adds a hit to the cache key
        """
        async with asafely(timeout=self.timeout):
            key_hits = await self._aget('keyhits') or {}
            if key not in key_hits:
                key_hits[key] = 0
            key_hits[key] += 1
            await self._aset('keyhits', key_hits)

    async def aadd_key_timestamp(self, key: str):
        """
        Adds a timestamp to the cache key
        """
        async with asafely(timeout=self.timeout):
            timestamps = await self._aget('timestamps') or {}
            timestamps[key] = time.perf_counter()
            await self._aset('timestamps', timestamps)

    async def aadd_hit(self):
        """
        Adds a hit to the cache
        """
        async with asafely(timeout=self.timeout):
            await self._aincr('hits')

    async def aencode_hit(self, value: Any, *args, **kwargs) -> bytes:
        """
        Encodes the hit
        """
        if self.hit_setter is not None:
            value = await run_as_coro(self.hit_setter, value, *args, **kwargs)
        return self.encode(value)

    async def adecode_hit(self, value: bytes, *args, **kwargs) -> Any:
        """
        Decodes the hit
        """
        value = self.decode(value)
        if self.hit_getter is not None:
            value = await run_as_coro(self.hit_getter, value, *args, **kwargs)
        return value

    async def acheck_cache_policies(self, key: str, *args, **kwargs) -> None:
        """
        Runs the cache policies
        """
        if await self.anum_keys <= self.cache_max_size:
            return
        num_keys = await self.anum_keys
        if self.verbose:
            logger.info(f'[{self.cache_field}] Cache Max Size Reached: {num_keys}/{self.cache_max_size}. Running Cache Policy: {self.cache_max_size_policy}')
        if self.cache_max_size_policy == CachePolicy.LRU:
            timestamps = await self.session.async_client.hget(self.cache_field, 'timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            await self.aclear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.LFU:
            key_hits = await self.session.async_client.hget(self.cache_field, 'keyhits') or {}
            keys_to_delete = sorted(key_hits, key=key_hits.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            await self.aclear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.FIFO:
            timestamps = await self.session.async_client.hget(self.cache_field, 'timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get, reverse=True)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            await self.aclear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.LIFO:
            timestamps = await self.session.async_client.hget(self.cache_field, 'timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            await self.aclear(keys_to_delete)
            return

    async def avalidate_cache_policies(self, key: str, *args, **kwargs) -> None:
        """
        Runs the cache policies
        """
        await self.aadd_hit()
        if self.cache_max_size is None:
            return
        await self.aadd_key_timestamp(key)
        await self.aadd_key_hit(key)
        await self.acheck_cache_policies(key, *args, **kwargs)

    async def ashould_not_retrieve(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the value should be retrieved
        which is based on the overwrite_if function
        """
        if self.overwrite_if is not None:
            return await run_as_coro(self.overwrite_if, *args, **kwargs)
        return False

    async def aretrieve(self, key: str, *args, **kwargs) -> Any:
        """
        Retrieves the value from the cache
        """
        if await self.ashould_not_retrieve(*args, **kwargs):
            if self.super_verbose:
                logger.info(f'[{self.cache_field}:{key}] Not Retrieving')
            return ENOVAL
        try:
            async with afail_after(self.timeout):
                if not await self.session.async_client.hexists(self.cache_field, key):
                    if self.super_verbose:
                        logger.info(f'[{self.cache_field}:{key}] Not Found')
                    return ENOVAL
                value = await self.session.async_client.hget(self.cache_field, key)
        except TimeoutError:
            if self.super_verbose:
                logger.error(f'[{self.cache_field}:{key}] Retrieve Timeout')
            return ENOVAL
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Retrieve Exception', error=e)
            return ENOVAL
        create_background_task(self.avalidate_cache_policies, key, *args, **kwargs)
        try:
            result = await self.adecode_hit(value, *args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Decode Exception', error=e)
        return ENOVAL

    async def aset(self, key: str, value: Any, *args, **kwargs) -> None:
        """
        Sets the value in the cache
        """
        try:
            async with afail_after(self.timeout):
                await self._aset(key, await self.aencode_hit(value, *args, **kwargs))
                if self.ttl:
                    await self.session.async_client.expire(self.cache_field, self.ttl)
        except TimeoutError:
            if self.super_verbose:
                logger.error(f'[{self.cache_field}:{key}] Set Timeout')
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Set Exception: {value}', error=e)

    async def aclear(self, keys: Union[str, List[str]]=None) -> Optional[int]:
        """
        Clears the cache
        """
        async with asafely(timeout=self.timeout):
            keys = keys or []
            if isinstance(keys, str):
                keys = [keys]
            return await self._aclear(*keys)

    def add_key_hit(self, key: str):
        """
        Adds a hit to the cache key
        """
        with safely(timeout=self.timeout):
            key_hits = self._get('keyhits') or {}
            if key not in key_hits:
                key_hits[key] = 0
            key_hits[key] += 1
            self._set('keyhits', key_hits)

    def add_key_timestamp(self, key: str):
        """
        Adds a timestamp to the cache key
        """
        with safely(timeout=self.timeout):
            timestamps = self._get('timestamps') or {}
            timestamps[key] = time.perf_counter()
            self._set('timestamps', timestamps)

    def add_hit(self):
        """
        Adds a hit to the cache
        """
        with safely(timeout=self.timeout):
            self._incr('hits')

    def encode_hit(self, value: Any, *args, **kwargs) -> bytes:
        """
        Encodes the hit
        """
        if self.hit_setter is not None:
            value = self.hit_setter(value, *args, **kwargs)
        return self.encode(value)

    def decode_hit(self, value: bytes, *args, **kwargs) -> Any:
        """
        Decodes the hit
        """
        value = self.decode(value)
        if self.hit_getter is not None:
            value = self.hit_getter(value, *args, **kwargs)
        return value

    def check_cache_policies(self, key: str, *args, **kwargs) -> None:
        """
        Runs the cache policies
        """
        if self.num_keys <= self.cache_max_size:
            return
        num_keys = self.num_keys
        if self.verbose:
            logger.info(f'[{self.cache_field}] Cache Max Size Reached: {num_keys}/{self.cache_max_size}. Running Cache Policy: {self.cache_max_size_policy}')
        if self.cache_max_size_policy == CachePolicy.LRU:
            timestamps = self._get('timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            self.clear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.LFU:
            key_hits = self._get('keyhits') or {}
            keys_to_delete = sorted(key_hits, key=key_hits.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            self.clear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.FIFO:
            timestamps = self._get('timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get, reverse=True)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            self.clear(keys_to_delete)
            return
        if self.cache_max_size_policy == CachePolicy.LIFO:
            timestamps = self._get('timestamps') or {}
            keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
            if key in keys_to_delete:
                keys_to_delete.remove(key)
            if self.verbose:
                logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
            self.clear(keys_to_delete)
            return

    def validate_cache_policies(self, key: str, *args, **kwargs) -> None:
        """
        Runs the cache policies
        """
        self.add_hit()
        if self.cache_max_size is None:
            return
        self.add_key_timestamp(key)
        self.add_key_hit(key)
        self.check_cache_policies(key, *args, **kwargs)

    def should_not_retrieve(self, *args, **kwargs) -> bool:
        """
        Returns whether or not the value should be retrieved
        which is based on the overwrite_if function
        """
        if self.overwrite_if is not None:
            return self.overwrite_if(*args, **kwargs)
        return False

    def retrieve(self, key: str, *args, **kwargs) -> Any:
        """
        Retrieves the value from the cache
        """
        if self.should_not_retrieve(*args, **kwargs):
            if self.super_verbose:
                logger.info(f'[{self.cache_field}:{key}] Not Retrieving')
            return ENOVAL
        try:
            with anyio.fail_after(self.timeout):
                if not self.session.client.hexists(self.cache_field, key):
                    if self.super_verbose:
                        logger.info(f'[{self.cache_field}:{key}] Not Found')
                    return ENOVAL
                value = self.session.client.hget(self.cache_field, key)
        except TimeoutError:
            if self.super_verbose:
                logger.error(f'[{self.cache_field}:{key}] Retrieve Timeout')
            return ENOVAL
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Retrieve Exception', error=e)
            return ENOVAL
        create_background_task(self.validate_cache_policies, key, *args, **kwargs)
        try:
            result = self.decode_hit(value, *args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Decode Exception', error=e)
        return ENOVAL

    def set(self, key: str, value: Any, *args, **kwargs) -> None:
        """
        Sets the value in the cache
        """
        try:
            with fail_after(self.timeout):
                self.session.client.hset(self.cache_field, key, self.encode_hit(value, *args, **kwargs))
                if self.ttl:
                    self.session.client.expire(self.cache_field, self.ttl)
        except TimeoutError:
            if self.super_verbose:
                logger.error(f'[{self.cache_field}:{key}] Set Timeout')
        except Exception as e:
            if self.verbose:
                logger.trace(f'[{self.cache_field}:{key}] Set Exception: {value}', error=e)

    def clear(self, keys: Union[str, List[str]]=None) -> Optional[int]:
        """
        Clears the cache
        """
        with safely(timeout=self.timeout):
            if keys:
                return self.session.client.hdel(self.cache_field, keys)
            else:
                return self.session.client.delete(self.cache_field)

    def validate_is_class_method(self, func: Callable):
        """
        Validates if the function is a class method
        """
        if self.is_class_method is not None:
            return
        self.is_class_method = hasattr(func, '__class__') and inspect.isclass(func.__class__) and isclassmethod(func)

    async def arun_post_init_hook(self, func: Callable, *args, **kwargs) -> None:
        """
        Runs the post init hook which fires once after the function is initialized
        """
        if not self.has_post_init_hook:
            return
        if self.has_ran_post_init_hook:
            return
        if self.verbose:
            logger.info(f'[{self.cache_field}] Running Post Init Hook')
        create_background_task(self.post_init_hook, func, *args, **kwargs)
        self.has_ran_post_init_hook = True

    async def arun_post_call_hook(self, result: Any, *args, is_hit: Optional[bool]=None, **kwargs) -> None:
        """
        Runs the post call hook which fires after the function is called
        """
        if not self.has_post_call_hook:
            return
        if self.super_verbose:
            logger.info(f'[{self.cache_field}] Running Post Call Hook')
        create_background_task(self.post_call_hook, result, *args, is_hit=is_hit, **kwargs)

    def run_post_init_hook(self, func: Callable, *args, **kwargs) -> None:
        """
        Runs the post init hook which fires once after the function is initialized
        """
        if not self.has_post_init_hook:
            return
        if self.has_ran_post_init_hook:
            return
        if self.verbose:
            logger.info(f'[{self.cache_field}] Running Post Init Hook')
        create_background_task(self.post_init_hook, func, *args, **kwargs)
        self.has_ran_post_init_hook = True

    def run_post_call_hook(self, result: Any, *args, is_hit: Optional[bool]=None, **kwargs) -> None:
        """
        Runs the post call hook which fires after the function is called
        """
        if not self.has_post_call_hook:
            return
        if self.super_verbose:
            logger.info(f'[{self.cache_field}] Running Post Call Hook')
        create_background_task(self.post_call_hook, result, *args, is_hit=is_hit, **kwargs)