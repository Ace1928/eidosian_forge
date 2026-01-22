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
def _start_proxies_if_needed(self, target_nodes) -> None:
    """Start a proxy on every node if it doesn't already exist."""
    for node_id, node_ip_address in target_nodes:
        if node_id in self._proxy_states:
            continue
        name = self._generate_actor_name(node_id=node_id)
        actor_proxy_wrapper = self._start_proxy(name=name, node_id=node_id, node_ip_address=node_ip_address)
        self._proxy_states[node_id] = ProxyState(actor_proxy_wrapper=actor_proxy_wrapper, actor_name=name, node_id=node_id, node_ip=node_ip_address, proxy_restart_count=self._proxy_restart_counts.get(node_id, 0), timer=self._timer)