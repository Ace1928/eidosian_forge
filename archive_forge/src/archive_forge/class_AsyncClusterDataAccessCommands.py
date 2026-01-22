import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class AsyncClusterDataAccessCommands(ClusterDataAccessCommands, AsyncDataAccessCommands):
    """
    A class for Redis Cluster Data Access Commands

    The class inherits from Redis's core DataAccessCommand class and do the
    required adjustments to work with cluster mode
    """

    async def scan_iter(self, match: Optional[PatternT]=None, count: Optional[int]=None, _type: Optional[str]=None, **kwargs) -> AsyncIterator:
        cursors, data = await self.scan(match=match, count=count, _type=_type, **kwargs)
        for value in data:
            yield value
        cursors = {name: cursor for name, cursor in cursors.items() if cursor != 0}
        if cursors:
            nodes = {name: self.get_node(node_name=name) for name in cursors.keys()}
            kwargs.pop('target_nodes', None)
            while cursors:
                for name, cursor in cursors.items():
                    cur, data = await self.scan(cursor=cursor, match=match, count=count, _type=_type, target_nodes=nodes[name], **kwargs)
                    for value in data:
                        yield value
                    cursors[name] = cur[name]
                cursors = {name: cursor for name, cursor in cursors.items() if cursor != 0}