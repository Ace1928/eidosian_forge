from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Tuple
import ray
from ray._private.ray_constants import AUTOSCALER_NAMESPACE, AUTOSCALER_V2_ENABLED_KEY
from ray._private.utils import binary_to_hex
from ray.autoscaler._private.autoscaler import AutoscalerSummary
from ray.autoscaler._private.node_provider_availability_tracker import (
from ray.autoscaler._private.util import LoadMetricsSummary, format_info_string
from ray.autoscaler.v2.schema import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
from ray.experimental.internal_kv import _internal_kv_get, _internal_kv_initialized
class ClusterStatusParser:

    @classmethod
    def from_get_cluster_status_reply(cls, proto: GetClusterStatusReply, stats: Stats) -> ClusterStatus:
        active_nodes, idle_nodes, failed_nodes = cls._parse_nodes(proto.cluster_resource_state)
        pending_nodes = cls._parse_pending(proto.autoscaling_state)
        pending_launches, failed_launches = cls._parse_launch_requests(proto.autoscaling_state)
        cluster_resource_usage = cls._parse_cluster_resource_usage(proto.cluster_resource_state)
        resource_demands = cls._parse_resource_demands(proto.cluster_resource_state)
        stats = cls._parse_stats(proto, stats)
        return ClusterStatus(active_nodes=active_nodes, idle_nodes=idle_nodes, pending_launches=pending_launches, failed_launches=failed_launches, pending_nodes=pending_nodes, failed_nodes=failed_nodes, cluster_resource_usage=cluster_resource_usage, resource_demands=resource_demands, stats=stats)

    @classmethod
    def _parse_stats(cls, reply: GetClusterStatusReply, stats: Stats) -> Stats:
        """
        Parse the stats from the get cluster status reply.
        Args:
            reply: the get cluster status reply
            stats: the stats
        Returns:
            stats: the parsed stats
        """
        stats = deepcopy(stats)
        stats.gcs_request_time_s = stats.gcs_request_time_s
        stats.autoscaler_version = str(reply.autoscaling_state.autoscaler_state_version)
        stats.cluster_resource_state_version = str(reply.cluster_resource_state.cluster_resource_state_version)
        return stats

    @classmethod
    def _parse_resource_demands(cls, state: ClusterResourceState) -> List[ResourceDemand]:
        """
        Parse the resource demands from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            resource_demands: the resource demands
        """
        task_actor_demand = []
        pg_demand = []
        constraint_demand = []
        for request_count in state.pending_resource_requests:
            demand = RayTaskActorDemand(bundles_by_count=[ResourceRequestByCount(request_count.request.resources_bundle, request_count.count)])
            task_actor_demand.append(demand)
        for gang_request in state.pending_gang_resource_requests:
            demand = PlacementGroupResourceDemand(bundles_by_count=cls._aggregate_resource_requests_by_shape(gang_request.requests), details=gang_request.details)
            pg_demand.append(demand)
        for constraint_request in state.cluster_resource_constraints:
            demand = ClusterConstraintDemand(bundles_by_count=[ResourceRequestByCount(bundle=dict(r.request.resources_bundle.items()), count=r.count) for r in constraint_request.min_bundles])
            constraint_demand.append(demand)
        return ResourceDemandSummary(ray_task_actor_demand=task_actor_demand, placement_group_demand=pg_demand, cluster_constraint_demand=constraint_demand)

    @classmethod
    def _aggregate_resource_requests_by_shape(cls, requests: List[ResourceRequest]) -> List[ResourceRequestByCount]:
        """
        Aggregate resource requests by shape.
        Args:
            requests: the list of resource requests
        Returns:
            resource_requests_by_count: the aggregated resource requests by count
        """
        resource_requests_by_count = defaultdict(int)
        for request in requests:
            bundle = frozenset(request.resources_bundle.items())
            resource_requests_by_count[bundle] += 1
        return [ResourceRequestByCount(dict(bundle), count) for bundle, count in resource_requests_by_count.items()]

    @classmethod
    def _parse_node_resource_usage(cls, node_state: NodeState, usage: Dict[str, ResourceUsage]) -> Dict[str, ResourceUsage]:
        """
        Parse the node resource usage from the node state.
        Args:
            node_state: the node state
            usage: the usage dict to be updated. This is a dict of
                {resource_name: ResourceUsage}
        Returns:
            usage: the updated usage dict
        """
        d = defaultdict(lambda: [0.0, 0.0])
        for resource_name, resource_total in node_state.total_resources.items():
            d[resource_name][1] += resource_total
            d[resource_name][0] += resource_total
        for resource_name, resource_available in node_state.available_resources.items():
            d[resource_name][0] -= resource_available
        for k, (used, total) in d.items():
            usage[k].resource_name = k
            usage[k].used += used
            usage[k].total += total
        return usage

    @classmethod
    def _parse_cluster_resource_usage(cls, state: ClusterResourceState) -> List[ResourceUsage]:
        """
        Parse the cluster resource usage from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            cluster_resource_usage: the cluster resource usage
        """
        cluster_resource_usage = defaultdict(ResourceUsage)
        for node_state in state.node_states:
            if node_state.status != NodeStatus.DEAD:
                cluster_resource_usage = cls._parse_node_resource_usage(node_state, cluster_resource_usage)
        return list(cluster_resource_usage.values())

    @classmethod
    def _parse_nodes(cls, state: ClusterResourceState) -> Tuple[List[NodeInfo], List[NodeInfo]]:
        """
        Parse the node info from the cluster resource state.
        Args:
            state: the cluster resource state
        Returns:
            active_nodes: the list of non-idle nodes
            idle_nodes: the list of idle nodes
            dead_nodes: the list of dead nodes
        """
        active_nodes = []
        dead_nodes = []
        idle_nodes = []
        for node_state in state.node_states:
            node_id = binary_to_hex(node_state.node_id)
            if len(node_state.ray_node_type_name) == 0:
                ray_node_type_name = f'node_{node_id}'
            else:
                ray_node_type_name = node_state.ray_node_type_name
            node_resource_usage = None
            failure_detail = None
            if node_state.status == NodeStatus.DEAD:
                failure_detail = NODE_DEATH_CAUSE_RAYLET_DIED
            else:
                usage = defaultdict(ResourceUsage)
                usage = cls._parse_node_resource_usage(node_state, usage)
                node_resource_usage = NodeUsage(usage=list(usage.values()), idle_time_ms=node_state.idle_duration_ms if node_state.status == NodeStatus.IDLE else 0)
            node_info = NodeInfo(instance_type_name=node_state.instance_type_name, node_status=NodeStatus.Name(node_state.status), node_id=binary_to_hex(node_state.node_id), ip_address=node_state.node_ip_address, ray_node_type_name=ray_node_type_name, instance_id=node_state.instance_id, resource_usage=node_resource_usage, failure_detail=failure_detail, node_activity=node_state.node_activity)
            if node_state.status == NodeStatus.DEAD:
                dead_nodes.append(node_info)
            elif node_state.status == NodeStatus.IDLE:
                idle_nodes.append(node_info)
            else:
                active_nodes.append(node_info)
        return (active_nodes, idle_nodes, dead_nodes)

    @classmethod
    def _parse_launch_requests(cls, state: AutoscalingState) -> Tuple[List[LaunchRequest], List[LaunchRequest]]:
        """
        Parse the launch requests from the autoscaling state.
        Args:
            state: the autoscaling state, empty if there's no autoscaling state
                being reported.
        Returns:
            pending_launches: the list of pending launches
            failed_launches: the list of failed launches
        """
        pending_launches = []
        for pending_request in state.pending_instance_requests:
            launch = LaunchRequest(instance_type_name=pending_request.instance_type_name, ray_node_type_name=pending_request.ray_node_type_name, count=pending_request.count, state=LaunchRequest.Status.PENDING, request_ts_s=pending_request.request_ts)
            pending_launches.append(launch)
        failed_launches = []
        for failed_request in state.failed_instance_requests:
            launch = LaunchRequest(instance_type_name=failed_request.instance_type_name, ray_node_type_name=failed_request.ray_node_type_name, count=failed_request.count, state=LaunchRequest.Status.FAILED, request_ts_s=failed_request.start_ts, details=failed_request.reason, failed_ts_s=failed_request.failed_ts)
            failed_launches.append(launch)
        return (pending_launches, failed_launches)

    @classmethod
    def _parse_pending(cls, state: AutoscalingState) -> List[NodeInfo]:
        """
        Parse the pending requests/nodes from the autoscaling state.
        Args:
            state: the autoscaling state, empty if there's no autoscaling state
                being reported.
        Returns:
            pending_nodes: the list of pending nodes
        """
        pending_nodes = []
        for pending_node in state.pending_instances:
            pending_nodes.append(NodeInfo(instance_type_name=pending_node.instance_type_name, ray_node_type_name=pending_node.ray_node_type_name, details=pending_node.details, instance_id=pending_node.instance_id, ip_address=pending_node.ip_address))
        return pending_nodes