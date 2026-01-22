import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class AsyncRedisClusterCommands(AsyncClusterMultiKeyCommands, AsyncClusterManagementCommands, AsyncACLCommands, AsyncClusterDataAccessCommands, AsyncScriptCommands, AsyncFunctionCommands, AsyncGearsCommands, AsyncModuleCommands, AsyncRedisModuleCommands):
    """
    A class for all Redis Cluster commands

    For key-based commands, the target node(s) will be internally determined
    by the keys' hash slot.
    Non-key-based commands can be executed with the 'target_nodes' argument to
    target specific nodes. By default, if target_nodes is not specified, the
    command will be executed on the default cluster node.

    :param :target_nodes: type can be one of the followings:
        - nodes flag: ALL_NODES, PRIMARIES, REPLICAS, RANDOM
        - 'ClusterNode'
        - 'list(ClusterNodes)'
        - 'dict(any:clusterNodes)'

    for example:
        r.cluster_info(target_nodes=RedisCluster.ALL_NODES)
    """