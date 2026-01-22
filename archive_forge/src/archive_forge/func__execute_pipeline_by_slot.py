import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def _execute_pipeline_by_slot(self, command: str, slots_to_args: Mapping[int, Iterable[EncodableT]]) -> List[Any]:
    read_from_replicas = self.read_from_replicas and command in READ_COMMANDS
    pipe = self.pipeline()
    [pipe.execute_command(command, *slot_args, target_nodes=[self.nodes_manager.get_node_from_slot(slot, read_from_replicas)]) for slot, slot_args in slots_to_args.items()]
    return pipe.execute()