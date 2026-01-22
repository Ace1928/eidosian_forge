import json
import logging
import os
import random
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import NodeId, ProxyStatus
from ray.serve._private.constants import (
from ray.serve._private.proxy import ProxyActor
from ray.serve._private.utils import Timer, TimerBase, format_actor_name
from ray.serve.config import DeploymentMode, HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig, ProxyDetails
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
class ProxyStateManager:
    """Manages all state for proxies in the system.

    This class is *not* thread safe, so any state-modifying methods should be
    called with a lock held.
    """

    def __init__(self, controller_name: str, config: HTTPOptions, head_node_id: str, cluster_node_info_cache: ClusterNodeInfoCache, logging_config: LoggingConfig, grpc_options: Optional[gRPCOptions]=None, proxy_actor_class: Type[ProxyActor]=ProxyActor, actor_proxy_wrapper_class: Type[ProxyWrapper]=ActorProxyWrapper, timer: TimerBase=Timer()):
        self.logging_config = logging_config
        self._controller_name = controller_name
        if config is not None:
            self._config = config
        else:
            self._config = HTTPOptions()
        self._grpc_options = grpc_options or gRPCOptions()
        self._proxy_states: Dict[NodeId, ProxyState] = dict()
        self._proxy_restart_counts: Dict[NodeId, int] = dict()
        self._head_node_id: str = head_node_id
        self._proxy_actor_class = proxy_actor_class
        self._actor_proxy_wrapper_class = actor_proxy_wrapper_class
        self._timer = timer
        self._cluster_node_info_cache = cluster_node_info_cache
        assert isinstance(head_node_id, str)

    def reconfiture_logging_config(self, logging_config: LoggingConfig):
        self.logging_config = logging_config

    def shutdown(self) -> None:
        for proxy_state in self._proxy_states.values():
            proxy_state.shutdown()

    def is_ready_for_shutdown(self) -> bool:
        """Return whether all proxies are shutdown.

        Iterate through all proxy states and check if all their proxy actors
        are shutdown.
        """
        return all((proxy_state.is_ready_for_shutdown() for proxy_state in self._proxy_states.values()))

    def get_config(self) -> HTTPOptions:
        return self._config

    def get_grpc_config(self) -> gRPCOptions:
        return self._grpc_options

    def get_proxy_handles(self) -> Dict[NodeId, ActorHandle]:
        return {node_id: state.actor_handle for node_id, state in self._proxy_states.items()}

    def get_proxy_names(self) -> Dict[NodeId, str]:
        return {node_id: state.actor_name for node_id, state in self._proxy_states.items()}

    def get_proxy_details(self) -> Dict[NodeId, ProxyDetails]:
        return {node_id: state.actor_details for node_id, state in self._proxy_states.items()}

    def update(self, proxy_nodes: Set[NodeId]=None):
        """Update the state of all proxies.

        Start proxies on all nodes if not already exist and stop the proxies on nodes
        that are no longer exist. Update all proxy states. Kill and restart
        unhealthy proxies.
        """
        if proxy_nodes is None:
            proxy_nodes = {self._head_node_id}
        else:
            proxy_nodes.add(self._head_node_id)
        target_nodes = self._get_target_nodes(proxy_nodes)
        target_node_ids = {node_id for node_id, _ in target_nodes}
        for node_id, proxy_state in self._proxy_states.items():
            draining = node_id not in target_node_ids
            proxy_state.update(draining)
        self._stop_proxies_if_needed()
        self._start_proxies_if_needed(target_nodes)

    def _get_target_nodes(self, proxy_nodes) -> List[Tuple[str, str]]:
        """Return the list of (node_id, ip_address) to deploy HTTP and gRPC servers
        on."""
        location = self._config.location
        if location == DeploymentMode.NoServer:
            return []
        target_nodes = [(node_id, ip_address) for node_id, ip_address in self._cluster_node_info_cache.get_alive_nodes() if node_id in proxy_nodes]
        if location == DeploymentMode.HeadOnly:
            nodes = [(node_id, ip_address) for node_id, ip_address in target_nodes if node_id == self._head_node_id]
            assert len(nodes) == 1, f'Head node not found! Head node id: {self._head_node_id}, all nodes: {target_nodes}.'
            return nodes
        return target_nodes

    def _generate_actor_name(self, node_id: str) -> str:
        return format_actor_name(SERVE_PROXY_NAME, self._controller_name, node_id)

    def _start_proxy(self, name: str, node_id: str, node_ip_address: str) -> ProxyWrapper:
        """Helper to start or reuse existing proxy and wrap in the proxy actor wrapper.

        Compute the HTTP port based on `TEST_WORKER_NODE_HTTP_PORT` env var and gRPC
        port based on `TEST_WORKER_NODE_GRPC_PORT` env var. Passed all the required
        variables into the proxy actor wrapper class and return the proxy actor wrapper.
        """
        port = self._config.port
        grpc_options = self._grpc_options
        if node_id != self._head_node_id and os.getenv('TEST_WORKER_NODE_HTTP_PORT') is not None:
            logger.warning(f'`TEST_WORKER_NODE_HTTP_PORT` env var is set. Using it for worker node {node_id}.')
            port = int(os.getenv('TEST_WORKER_NODE_HTTP_PORT'))
        if node_id != self._head_node_id and os.getenv('TEST_WORKER_NODE_GRPC_PORT') is not None:
            logger.warning(f'`TEST_WORKER_NODE_GRPC_PORT` env var is set. Using it for worker node {node_id}.{int(os.getenv('TEST_WORKER_NODE_GRPC_PORT'))}')
            grpc_options.port = int(os.getenv('TEST_WORKER_NODE_GRPC_PORT'))
        return self._actor_proxy_wrapper_class(logging_config=self.logging_config, config=self._config, grpc_options=grpc_options, controller_name=self._controller_name, name=name, node_id=node_id, node_ip_address=node_ip_address, port=port, proxy_actor_class=self._proxy_actor_class)

    def _start_proxies_if_needed(self, target_nodes) -> None:
        """Start a proxy on every node if it doesn't already exist."""
        for node_id, node_ip_address in target_nodes:
            if node_id in self._proxy_states:
                continue
            name = self._generate_actor_name(node_id=node_id)
            actor_proxy_wrapper = self._start_proxy(name=name, node_id=node_id, node_ip_address=node_ip_address)
            self._proxy_states[node_id] = ProxyState(actor_proxy_wrapper=actor_proxy_wrapper, actor_name=name, node_id=node_id, node_ip=node_ip_address, proxy_restart_count=self._proxy_restart_counts.get(node_id, 0), timer=self._timer)

    def _stop_proxies_if_needed(self) -> bool:
        """Removes proxy actors.

        Removes proxy actors from any nodes that no longer exist or unhealthy proxy.
        """
        alive_node_ids = self._cluster_node_info_cache.get_alive_node_ids()
        to_stop = []
        for node_id, proxy_state in self._proxy_states.items():
            if node_id not in alive_node_ids:
                logger.info(f"Removing proxy on removed node '{node_id}'.")
                to_stop.append(node_id)
            elif proxy_state.status == ProxyStatus.UNHEALTHY:
                logger.info(f"Proxy on node '{node_id}' UNHEALTHY. Shutting down the unhealthy proxy and starting a new one.")
                to_stop.append(node_id)
            elif proxy_state.status == ProxyStatus.DRAINED:
                logger.info(f"Removing drained proxy on node '{node_id}'.")
                to_stop.append(node_id)
        for node_id in to_stop:
            proxy_state = self._proxy_states.pop(node_id)
            self._proxy_restart_counts[node_id] = proxy_state.proxy_restart_count + 1
            proxy_state.shutdown()