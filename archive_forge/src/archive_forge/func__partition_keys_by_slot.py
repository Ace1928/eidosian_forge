import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def _partition_keys_by_slot(self, keys: Iterable[KeyT]) -> Dict[int, List[KeyT]]:
    """Split keys into a dictionary that maps a slot to a list of keys."""
    slots_to_keys = {}
    for key in keys:
        slot = key_slot(self.encoder.encode(key))
        slots_to_keys.setdefault(slot, []).append(key)
    return slots_to_keys