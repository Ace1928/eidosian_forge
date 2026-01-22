import asyncio
from typing import (
from aiokeydb.v1.compat import Literal
from aiokeydb.v1.crc import key_slot
from aiokeydb.v1.exceptions import KeyDBClusterException, KeyDBError
from aiokeydb.v1.typing import (
from aiokeydb.v1.commands.core import (
from aiokeydb.v1.commands.helpers import list_or_args
from aiokeydb.v1.commands.redismodules import RedisModuleCommands
class KeyDBClusterCommands(ClusterMultiKeyCommands, ClusterManagementCommands, ACLCommands, PubSubCommands, ClusterDataAccessCommands, ScriptCommands, FunctionCommands, RedisModuleCommands):
    """
    A class for all KeyDB Cluster commands

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
        r.cluster_info(target_nodes=KeyDBCluster.ALL_NODES)
    """