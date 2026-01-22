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
def _drain_check(self):
    """Check whether the proxy actor is drained or not."""
    assert self._status == ProxyStatus.DRAINING
    if self._actor_proxy_wrapper.is_draining:
        try:
            drained_call_status = self._actor_proxy_wrapper.is_drained()
            if drained_call_status == ProxyWrapperCallStatus.FINISHED_SUCCEED:
                self.set_status(ProxyStatus.DRAINED)
        except Exception as e:
            logger.warning(f'Drain check for proxy {self._actor_name} failed: {e}.')
    elif self._timer.time() - self._last_drain_check_time > PROXY_DRAIN_CHECK_PERIOD_S:
        self._last_drain_check_time = self._timer.time()
        self._actor_proxy_wrapper.start_new_drained_check()