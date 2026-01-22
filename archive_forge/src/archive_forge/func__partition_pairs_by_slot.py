import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def _partition_pairs_by_slot(self, mapping: Mapping[AnyKeyT, EncodableT]) -> Dict[int, List[EncodableT]]:
    """Split pairs into a dictionary that maps a slot to a list of pairs."""
    slots_to_pairs = {}
    for pair in mapping.items():
        slot = key_slot(self.encoder.encode(pair[0]))
        slots_to_pairs.setdefault(slot, []).extend(pair)
    return slots_to_pairs