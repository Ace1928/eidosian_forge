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
class ClusterStatusFormatter:
    """
    A formatter to format the ClusterStatus into a string.

    TODO(rickyx): We right now parse the ClusterStatus to the legacy format
    by using the `format_info_string`.
    In the future, we should refactor the `format_info_string` to directly format
    the ClusterStatus into a string as we migrate eventually away from v1.

    """

    @classmethod
    def format(cls, data: ClusterStatus, verbose: bool=False) -> str:
        lm_summary = cls._parse_lm_summary(data)
        autoscaler_summary = cls._parse_autoscaler_summary(data)
        return format_info_string(lm_summary, autoscaler_summary, time=datetime.fromtimestamp(data.stats.request_ts_s), gcs_request_time=data.stats.gcs_request_time_s, non_terminated_nodes_time=data.stats.none_terminated_node_request_time_s, autoscaler_update_time=data.stats.autoscaler_iteration_time_s, verbose=verbose)

    @classmethod
    def _parse_autoscaler_summary(cls, data: ClusterStatus) -> AutoscalerSummary:
        active_nodes = _count_by(data.active_nodes, 'ray_node_type_name')
        idle_nodes = _count_by(data.idle_nodes, 'ray_node_type_name')
        pending_launches = _count_by(data.pending_launches, 'ray_node_type_name')
        pending_nodes = []
        for node in data.pending_nodes:
            pending_nodes.append((node.ip_address, node.ray_node_type_name, node.details))
        failed_nodes = []
        for node in data.failed_nodes:
            failed_nodes.append((node.ip_address, node.ray_node_type_name))
        node_type_mapping = {}
        for node in chain(data.active_nodes, data.idle_nodes):
            node_type_mapping[node.ip_address] = node.ray_node_type_name
        node_availabilities = {}
        for failed_launch in data.failed_launches:
            node_availabilities[failed_launch.ray_node_type_name] = NodeAvailabilityRecord(node_type=failed_launch.ray_node_type_name, is_available=False, last_checked_timestamp=failed_launch.request_ts_s, unavailable_node_information=UnavailableNodeInformation(category='LaunchFailed', description=failed_launch.details))
        node_availabilities = NodeAvailabilitySummary(node_availabilities=node_availabilities)
        node_activities = {node.node_id: (node.ip_address, node.node_activity) for node in data.active_nodes}
        return AutoscalerSummary(active_nodes=active_nodes, idle_nodes=idle_nodes, pending_launches=pending_launches, pending_nodes=pending_nodes, failed_nodes=failed_nodes, pending_resources={}, node_type_mapping=node_type_mapping, node_availability_summary=node_availabilities, node_activities=node_activities)

    @classmethod
    def _parse_lm_summary(cls, data: ClusterStatus) -> LoadMetricsSummary:
        usage = {u.resource_name: (u.used, u.total) for u in data.cluster_resource_usage}
        resource_demands = []
        for demand in data.resource_demands.ray_task_actor_demand:
            for bundle_by_count in demand.bundles_by_count:
                resource_demands.append((bundle_by_count.bundle, bundle_by_count.count))
        pg_demand = []
        pg_demand_strs = []
        pg_demand_str_to_demand = {}
        for pg_demand in data.resource_demands.placement_group_demand:
            s = pg_demand.strategy + '|' + pg_demand.state
            pg_demand_strs.append(s)
            pg_demand_str_to_demand[s] = pg_demand
        pg_freqs = Counter(pg_demand_strs)
        pg_demand = [({'strategy': pg_demand_str_to_demand[pg_str].strategy, 'bundles': [(bundle_count.bundle, bundle_count.count) for bundle_count in pg_demand_str_to_demand[pg_str].bundles_by_count]}, freq) for pg_str, freq in pg_freqs.items()]
        request_demand = [(bc.bundle, bc.count) for constraint_demand in data.resource_demands.cluster_constraint_demand for bc in constraint_demand.bundles_by_count]
        usage_by_node = {}
        node_type_mapping = {}
        idle_time_map = {}
        for node in chain(data.active_nodes, data.idle_nodes):
            usage_by_node[node.node_id] = {u.resource_name: (u.used, u.total) for u in node.resource_usage.usage}
            node_type_mapping[node.node_id] = node.ray_node_type_name
            idle_time_map[node.node_id] = node.resource_usage.idle_time_ms
        return LoadMetricsSummary(usage=usage, resource_demand=resource_demands, pg_demand=pg_demand, request_demand=request_demand, node_types=None, usage_by_node=usage_by_node, node_type_mapping=node_type_mapping, idle_time_map=idle_time_map)