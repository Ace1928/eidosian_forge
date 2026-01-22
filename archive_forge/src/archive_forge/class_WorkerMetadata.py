import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
@dataclass
class WorkerMetadata:
    """Metadata for each worker/actor.

    This information is expected to stay the same throughout the lifetime of
    actor.

    Args:
        node_id: ID of the node this worker is on.
        node_ip: IP address of the node this worker is on.
        hostname: Hostname that this worker is on.
        resource_ids: Map of accelerator resources
        ("GPU", "neuron_cores", ..) to their IDs.
        pid: Process ID of this worker.
    """
    node_id: str
    node_ip: str
    hostname: str
    resource_ids: Dict[str, List[str]]
    pid: int