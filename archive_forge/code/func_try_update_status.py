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
def try_update_status(self, status: ProxyStatus):
    """Try update with the new status and only update when the conditions are met.

        Status will only set to UNHEALTHY after PROXY_HEALTH_CHECK_UNHEALTHY_THRESHOLD
        consecutive failures. A warning will be logged when the status is set to
        UNHEALTHY. Also, when status is set to HEALTHY, we will reset
        self._consecutive_health_check_failures to 0.
        """
    if status == ProxyStatus.UNHEALTHY:
        self._consecutive_health_check_failures += 1
        if self._consecutive_health_check_failures < PROXY_HEALTH_CHECK_UNHEALTHY_THRESHOLD:
            return
    if status != ProxyStatus.UNHEALTHY:
        self._consecutive_health_check_failures = 0
    self.set_status(status=status)
    if status == ProxyStatus.UNHEALTHY:
        logger.warning(f'Proxy {self._actor_name} failed the health check {self._consecutive_health_check_failures} times in a row, marking it unhealthy.')