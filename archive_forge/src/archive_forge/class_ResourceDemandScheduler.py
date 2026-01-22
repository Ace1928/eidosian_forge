import copy
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from ray._private.protobuf_compat import message_to_dict
from ray.autoscaler._private.resource_demand_scheduler import UtilizationScore
from ray.autoscaler.v2.schema import NodeType
from ray.autoscaler.v2.utils import is_pending, resource_requests_by_count
from ray.core.generated.autoscaler_pb2 import (
from ray.core.generated.instance_manager_pb2 import Instance
class ResourceDemandScheduler(IResourceScheduler):
    """
    A "simple" resource scheduler that schedules resource requests based on the
    following rules:
        1. Enforce the minimal count of nodes for each worker node type.
        2. Enforce the cluster resource constraints.
        3. Schedule the gang resource requests.
        4. Schedule the tasks/actor resource requests
    """

    @dataclass
    class ScheduleContext:
        """
        Encapsulates the context for processing one scheduling request.

        This exposes functions to read and write the scheduling nodes, to prevent
        accidental modification of the internal state.
        """
        _cluster_config: ClusterConfig
        _nodes: List[SchedulingNode] = field(default_factory=list)
        _node_type_available: Dict[NodeType, int] = field(default_factory=dict)

        def __init__(self, nodes: List[SchedulingNode], node_type_available: Dict[NodeType, int], cluster_config: ClusterConfig):
            self._nodes = nodes
            self._node_type_available = node_type_available
            self._cluster_config = cluster_config

        @classmethod
        def from_schedule_request(cls, req: SchedulingRequest) -> 'ResourceDemandScheduler.ScheduleContext':
            """
            Create a schedule context from a schedule request.
            It will populate the context with the existing nodes and the available node
            types from the config.

            Args:
                req: The scheduling request. The caller should make sure the
                    request is valid.
            """
            nodes = []
            for node in req.current_nodes:
                nodes.append(SchedulingNode(node_type=node.ray_node_type_name, total_resources=dict(node.total_resources), available_resources=dict(node.available_resources), labels=dict(node.dynamic_labels), status=SchedulingNodeStatus.RUNNING))
            cluster_config = req.cluster_config
            for instance in req.current_instances:
                if not is_pending(instance):
                    continue
                node_config = cluster_config.node_type_configs[instance.ray_node_type_name]
                nodes.append(SchedulingNode.from_node_config(node_config, status=SchedulingNodeStatus.PENDING))
            node_type_available = cls._compute_available_node_types(nodes, req.cluster_config)
            return cls(nodes=nodes, node_type_available=node_type_available, cluster_config=req.cluster_config)

        @staticmethod
        def _compute_available_node_types(nodes: List[SchedulingNode], cluster_config: ClusterConfig) -> Dict[NodeType, int]:
            """
            Compute the number of nodes by node types available for launching based on
            the max number of workers in the config.
            Args:
                nodes: The current existing nodes.
                cluster_config: The cluster instances config.
            Returns:
                A dict of node types and the number of nodes available for launching.
            """
            node_type_available: Dict[NodeType, int] = defaultdict(int)
            node_type_existing: Dict[NodeType, int] = defaultdict(int)
            for node in nodes:
                node_type_existing[node.node_type] += 1
            for node_type, node_type_config in cluster_config.node_type_configs.items():
                node_type_available[node_type] = node_type_config.max_workers - node_type_existing.get(node_type, 0)
            return node_type_available

        def get_nodes(self) -> List[SchedulingNode]:
            return copy.deepcopy(self._nodes)

        def get_cluster_shape(self) -> Dict[NodeType, int]:
            cluster_shape = defaultdict(int)
            for node in self._nodes:
                cluster_shape[node.node_type] += 1
            return cluster_shape

        def update(self, new_nodes: List[SchedulingNode]) -> None:
            """
            Update the context with the new nodes.
            """
            self._nodes = new_nodes
            self._node_type_available = self._compute_available_node_types(self._nodes, self._cluster_config)

        def get_cluster_config(self) -> ClusterConfig:
            return self._cluster_config

        def __str__(self) -> str:
            return 'ScheduleContext({} nodes, node_type_available={}): {}'.format(len(self._nodes), self._node_type_available, self._nodes)

    def schedule(self, request: SchedulingRequest) -> SchedulingReply:
        self._init_context(request)
        self._enforce_min_workers()
        infeasible_constraints = self._enforce_resource_constraints(request.cluster_resource_constraints)
        infeasible_gang_requests = self._sched_gang_resource_requests(request.gang_resource_requests)
        infeasible_requests = self._sched_resource_requests(request.resource_requests)
        reply = SchedulingReply(infeasible_resource_requests=resource_requests_by_count(infeasible_requests), infeasible_gang_resource_requests=infeasible_gang_requests, infeasible_cluster_resource_constraints=infeasible_constraints, target_cluster_shape=self._ctx.get_cluster_shape())
        return reply

    def _init_context(self, request: SchedulingRequest) -> None:
        self._ctx = self.ScheduleContext.from_schedule_request(request)

    def _enforce_min_workers(self) -> None:
        """
        Enforce the minimal count of nodes for each worker node type.
        """
        count_by_node_type = self._ctx.get_cluster_shape()
        logger.debug('Enforcing min workers: {}'.format(self._ctx))
        new_nodes = []
        for node_type, node_type_config in self._ctx.get_cluster_config().node_type_configs.items():
            cur_count = count_by_node_type.get(node_type, 0)
            min_count = node_type_config.min_workers
            if cur_count < min_count:
                new_nodes.extend([SchedulingNode.from_node_config(copy.deepcopy(node_type_config), status=SchedulingNodeStatus.TO_LAUNCH)] * (min_count - cur_count))
        self._ctx.update(new_nodes + self._ctx.get_nodes())
        logger.debug('After enforced min workers: {}'.format(self._ctx))

    def _enforce_resource_constraints(self, constraints: List[ClusterResourceConstraint]) -> List[ClusterResourceConstraint]:
        """
        Enforce the cluster resource constraints.

        Args:
            constraints: The cluster resource constraints.

        Returns:
            A list of infeasible constraints.

        Notes:
            It's different from the other scheduling functions since it doesn't actually
        schedule any resource requests. Instead, it asks if the cluster could be
        upscale to a certain shape to fulfill the constraints.
        """
        return []

    def _sched_resource_requests(self, requests_by_count: List[ResourceRequestByCount]) -> List[ResourceRequest]:
        """
        Schedule the resource requests.

        Args:
            requests_by_count: The resource requests.

        Returns:
            A list of infeasible resource requests.
        """
        return []

    def _sched_gang_resource_requests(self, gang_requests: List[GangResourceRequest]) -> List[GangResourceRequest]:
        """
        Schedule the gang resource requests.

        These requests should be scheduled atomically, i.e. either all of the resources
        requests in a gang request are scheduled or none of them are scheduled.

        Args:
            gang_requests: The gang resource requests.

        Returns:
            A list of infeasible gang resource requests.
        """
        return []