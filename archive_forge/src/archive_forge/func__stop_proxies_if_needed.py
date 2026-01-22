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