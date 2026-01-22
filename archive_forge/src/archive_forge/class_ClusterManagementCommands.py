import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class ClusterManagementCommands(ManagementCommands):
    """
    A class for Redis Cluster management commands

    The class inherits from Redis's core ManagementCommands class and do the
    required adjustments to work with cluster mode
    """

    def slaveof(self, *args, **kwargs) -> NoReturn:
        """
        Make the server a replica of another instance, or promote it as master.

        For more information see https://redis.io/commands/slaveof
        """
        raise RedisClusterException('SLAVEOF is not supported in cluster mode')

    def replicaof(self, *args, **kwargs) -> NoReturn:
        """
        Make the server a replica of another instance, or promote it as master.

        For more information see https://redis.io/commands/replicaof
        """
        raise RedisClusterException('REPLICAOF is not supported in cluster mode')

    def swapdb(self, *args, **kwargs) -> NoReturn:
        """
        Swaps two Redis databases.

        For more information see https://redis.io/commands/swapdb
        """
        raise RedisClusterException('SWAPDB is not supported in cluster mode')

    def cluster_myid(self, target_node: 'TargetNodesT') -> ResponseT:
        """
        Returns the node's id.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information check https://redis.io/commands/cluster-myid/
        """
        return self.execute_command('CLUSTER MYID', target_nodes=target_node)

    def cluster_addslots(self, target_node: 'TargetNodesT', *slots: EncodableT) -> ResponseT:
        """
        Assign new hash slots to receiving node. Sends to specified node.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-addslots
        """
        return self.execute_command('CLUSTER ADDSLOTS', *slots, target_nodes=target_node)

    def cluster_addslotsrange(self, target_node: 'TargetNodesT', *slots: EncodableT) -> ResponseT:
        """
        Similar to the CLUSTER ADDSLOTS command.
        The difference between the two commands is that ADDSLOTS takes a list of slots
        to assign to the node, while ADDSLOTSRANGE takes a list of slot ranges
        (specified by start and end slots) to assign to the node.

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-addslotsrange
        """
        return self.execute_command('CLUSTER ADDSLOTSRANGE', *slots, target_nodes=target_node)

    def cluster_countkeysinslot(self, slot_id: int) -> ResponseT:
        """
        Return the number of local keys in the specified hash slot
        Send to node based on specified slot_id

        For more information see https://redis.io/commands/cluster-countkeysinslot
        """
        return self.execute_command('CLUSTER COUNTKEYSINSLOT', slot_id)

    def cluster_count_failure_report(self, node_id: str) -> ResponseT:
        """
        Return the number of failure reports active for a given node
        Sends to a random node

        For more information see https://redis.io/commands/cluster-count-failure-reports
        """
        return self.execute_command('CLUSTER COUNT-FAILURE-REPORTS', node_id)

    def cluster_delslots(self, *slots: EncodableT) -> List[bool]:
        """
        Set hash slots as unbound in the cluster.
        It determines by it self what node the slot is in and sends it there

        Returns a list of the results for each processed slot.

        For more information see https://redis.io/commands/cluster-delslots
        """
        return [self.execute_command('CLUSTER DELSLOTS', slot) for slot in slots]

    def cluster_delslotsrange(self, *slots: EncodableT) -> ResponseT:
        """
        Similar to the CLUSTER DELSLOTS command.
        The difference is that CLUSTER DELSLOTS takes a list of hash slots to remove
        from the node, while CLUSTER DELSLOTSRANGE takes a list of slot ranges to remove
        from the node.

        For more information see https://redis.io/commands/cluster-delslotsrange
        """
        return self.execute_command('CLUSTER DELSLOTSRANGE', *slots)

    def cluster_failover(self, target_node: 'TargetNodesT', option: Optional[str]=None) -> ResponseT:
        """
        Forces a slave to perform a manual failover of its master
        Sends to specified node

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-failover
        """
        if option:
            if option.upper() not in ['FORCE', 'TAKEOVER']:
                raise RedisError(f'Invalid option for CLUSTER FAILOVER command: {option}')
            else:
                return self.execute_command('CLUSTER FAILOVER', option, target_nodes=target_node)
        else:
            return self.execute_command('CLUSTER FAILOVER', target_nodes=target_node)

    def cluster_info(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Provides info about Redis Cluster node state.
        The command will be sent to a random node in the cluster if no target
        node is specified.

        For more information see https://redis.io/commands/cluster-info
        """
        return self.execute_command('CLUSTER INFO', target_nodes=target_nodes)

    def cluster_keyslot(self, key: str) -> ResponseT:
        """
        Returns the hash slot of the specified key
        Sends to random node in the cluster

        For more information see https://redis.io/commands/cluster-keyslot
        """
        return self.execute_command('CLUSTER KEYSLOT', key)

    def cluster_meet(self, host: str, port: int, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Force a node cluster to handshake with another node.
        Sends to specified node.

        For more information see https://redis.io/commands/cluster-meet
        """
        return self.execute_command('CLUSTER MEET', host, port, target_nodes=target_nodes)

    def cluster_nodes(self) -> ResponseT:
        """
        Get Cluster config for the node.
        Sends to random node in the cluster

        For more information see https://redis.io/commands/cluster-nodes
        """
        return self.execute_command('CLUSTER NODES')

    def cluster_replicate(self, target_nodes: 'TargetNodesT', node_id: str) -> ResponseT:
        """
        Reconfigure a node as a slave of the specified master node

        For more information see https://redis.io/commands/cluster-replicate
        """
        return self.execute_command('CLUSTER REPLICATE', node_id, target_nodes=target_nodes)

    def cluster_reset(self, soft: bool=True, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Reset a Redis Cluster node

        If 'soft' is True then it will send 'SOFT' argument
        If 'soft' is False then it will send 'HARD' argument

        For more information see https://redis.io/commands/cluster-reset
        """
        return self.execute_command('CLUSTER RESET', b'SOFT' if soft else b'HARD', target_nodes=target_nodes)

    def cluster_save_config(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Forces the node to save cluster state on disk

        For more information see https://redis.io/commands/cluster-saveconfig
        """
        return self.execute_command('CLUSTER SAVECONFIG', target_nodes=target_nodes)

    def cluster_get_keys_in_slot(self, slot: int, num_keys: int) -> ResponseT:
        """
        Returns the number of keys in the specified cluster slot

        For more information see https://redis.io/commands/cluster-getkeysinslot
        """
        return self.execute_command('CLUSTER GETKEYSINSLOT', slot, num_keys)

    def cluster_set_config_epoch(self, epoch: int, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Set the configuration epoch in a new node

        For more information see https://redis.io/commands/cluster-set-config-epoch
        """
        return self.execute_command('CLUSTER SET-CONFIG-EPOCH', epoch, target_nodes=target_nodes)

    def cluster_setslot(self, target_node: 'TargetNodesT', node_id: str, slot_id: int, state: str) -> ResponseT:
        """
        Bind an hash slot to a specific node

        :target_node: 'ClusterNode'
            The node to execute the command on

        For more information see https://redis.io/commands/cluster-setslot
        """
        if state.upper() in ('IMPORTING', 'NODE', 'MIGRATING'):
            return self.execute_command('CLUSTER SETSLOT', slot_id, state, node_id, target_nodes=target_node)
        elif state.upper() == 'STABLE':
            raise RedisError('For "stable" state please use cluster_setslot_stable')
        else:
            raise RedisError(f'Invalid slot state: {state}')

    def cluster_setslot_stable(self, slot_id: int) -> ResponseT:
        """
        Clears migrating / importing state from the slot.
        It determines by it self what node the slot is in and sends it there.

        For more information see https://redis.io/commands/cluster-setslot
        """
        return self.execute_command('CLUSTER SETSLOT', slot_id, 'STABLE')

    def cluster_replicas(self, node_id: str, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Provides a list of replica nodes replicating from the specified primary
        target node.

        For more information see https://redis.io/commands/cluster-replicas
        """
        return self.execute_command('CLUSTER REPLICAS', node_id, target_nodes=target_nodes)

    def cluster_slots(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Get array of Cluster slot to node mappings

        For more information see https://redis.io/commands/cluster-slots
        """
        return self.execute_command('CLUSTER SLOTS', target_nodes=target_nodes)

    def cluster_shards(self, target_nodes=None):
        """
        Returns details about the shards of the cluster.

        For more information see https://redis.io/commands/cluster-shards
        """
        return self.execute_command('CLUSTER SHARDS', target_nodes=target_nodes)

    def cluster_myshardid(self, target_nodes=None):
        """
        Returns the shard ID of the node.

        For more information see https://redis.io/commands/cluster-myshardid/
        """
        return self.execute_command('CLUSTER MYSHARDID', target_nodes=target_nodes)

    def cluster_links(self, target_node: 'TargetNodesT') -> ResponseT:
        """
        Each node in a Redis Cluster maintains a pair of long-lived TCP link with each
        peer in the cluster: One for sending outbound messages towards the peer and one
        for receiving inbound messages from the peer.

        This command outputs information of all such peer links as an array.

        For more information see https://redis.io/commands/cluster-links
        """
        return self.execute_command('CLUSTER LINKS', target_nodes=target_node)

    def cluster_flushslots(self, target_nodes: Optional['TargetNodesT']=None) -> None:
        raise NotImplementedError('CLUSTER FLUSHSLOTS is intentionally not implemented in the client.')

    def cluster_bumpepoch(self, target_nodes: Optional['TargetNodesT']=None) -> None:
        raise NotImplementedError('CLUSTER BUMPEPOCH is intentionally not implemented in the client.')

    def readonly(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Enables read queries.
        The command will be sent to the default cluster node if target_nodes is
        not specified.

        For more information see https://redis.io/commands/readonly
        """
        if target_nodes == 'replicas' or target_nodes == 'all':
            self.read_from_replicas = True
        return self.execute_command('READONLY', target_nodes=target_nodes)

    def readwrite(self, target_nodes: Optional['TargetNodesT']=None) -> ResponseT:
        """
        Disables read queries.
        The command will be sent to the default cluster node if target_nodes is
        not specified.

        For more information see https://redis.io/commands/readwrite
        """
        self.read_from_replicas = False
        return self.execute_command('READWRITE', target_nodes=target_nodes)

    def gears_refresh_cluster(self, **kwargs) -> ResponseT:
        """
        On an OSS cluster, before executing any gears function, you must call this command. # noqa
        """
        return self.execute_command('REDISGEARS_2.REFRESHCLUSTER', **kwargs)