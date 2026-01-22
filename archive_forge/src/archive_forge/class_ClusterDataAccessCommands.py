import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class ClusterDataAccessCommands(DataAccessCommands):
    """
    A class for Redis Cluster Data Access Commands

    The class inherits from Redis's core DataAccessCommand class and do the
    required adjustments to work with cluster mode
    """

    def stralgo(self, algo: Literal['LCS'], value1: KeyT, value2: KeyT, specific_argument: Union[Literal['strings'], Literal['keys']]='strings', len: bool=False, idx: bool=False, minmatchlen: Optional[int]=None, withmatchlen: bool=False, **kwargs) -> ResponseT:
        """
        Implements complex algorithms that operate on strings.
        Right now the only algorithm implemented is the LCS algorithm
        (longest common substring). However new algorithms could be
        implemented in the future.

        ``algo`` Right now must be LCS
        ``value1`` and ``value2`` Can be two strings or two keys
        ``specific_argument`` Specifying if the arguments to the algorithm
        will be keys or strings. strings is the default.
        ``len`` Returns just the len of the match.
        ``idx`` Returns the match positions in each string.
        ``minmatchlen`` Restrict the list of matches to the ones of a given
        minimal length. Can be provided only when ``idx`` set to True.
        ``withmatchlen`` Returns the matches with the len of the match.
        Can be provided only when ``idx`` set to True.

        For more information see https://redis.io/commands/stralgo
        """
        target_nodes = kwargs.pop('target_nodes', None)
        if specific_argument == 'strings' and target_nodes is None:
            target_nodes = 'default-node'
        kwargs.update({'target_nodes': target_nodes})
        return super().stralgo(algo, value1, value2, specific_argument, len, idx, minmatchlen, withmatchlen, **kwargs)

    def scan_iter(self, match: Optional[PatternT]=None, count: Optional[int]=None, _type: Optional[str]=None, **kwargs) -> Iterator:
        cursors, data = self.scan(match=match, count=count, _type=_type, **kwargs)
        yield from data
        cursors = {name: cursor for name, cursor in cursors.items() if cursor != 0}
        if cursors:
            nodes = {name: self.get_node(node_name=name) for name in cursors.keys()}
            kwargs.pop('target_nodes', None)
            while cursors:
                for name, cursor in cursors.items():
                    cur, data = self.scan(cursor=cursor, match=match, count=count, _type=_type, target_nodes=nodes[name], **kwargs)
                    yield from data
                    cursors[name] = cur[name]
                cursors = {name: cursor for name, cursor in cursors.items() if cursor != 0}