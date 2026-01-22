import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
from .common import NodeIdStr
from ray.data._internal.execution.util import memory_string
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
@dataclass
class ExecutionOptions:
    """Common options for execution.

    Some options may not be supported on all executors (e.g., resource limits).

    Attributes:
        resource_limits: Set a soft limit on the resource usage during execution.
            This is not supported in bulk execution mode. Autodetected by default.
        exclude_resources: Amount of resources to exclude from Ray Data.
            Set this if you have other workloads running on the same cluster.
            Note,
            - If using Ray Data with Ray Train, training resources will be
            automatically excluded.
            - For each resource type, resource_limits and exclude_resources can
            not be both set.
        locality_with_output: Set this to prefer running tasks on the same node as the
            output node (node driving the execution). It can also be set to a list of
            node ids to spread the outputs across those nodes. Off by default.
        preserve_order: Set this to preserve the ordering between blocks processed by
            operators under the streaming executor. The bulk executor always preserves
            order. Off by default.
        actor_locality_enabled: Whether to enable locality-aware task dispatch to
            actors (on by default). This parameter applies to both stateful map and
            streaming_split operations.
        verbose_progress: Whether to report progress individually per operator. By
            default, only AllToAll operators and global progress is reported. This
            option is useful for performance debugging. Off by default.
    """
    resource_limits: ExecutionResources = field(default_factory=ExecutionResources)
    exclude_resources: ExecutionResources = field(default_factory=lambda: ExecutionResources(cpu=0, gpu=0, object_store_memory=0))
    locality_with_output: Union[bool, List[NodeIdStr]] = False
    preserve_order: bool = False
    actor_locality_enabled: bool = True
    verbose_progress: bool = bool(int(os.environ.get('RAY_DATA_VERBOSE_PROGRESS', '0')))

    def validate(self) -> None:
        """Validate the options."""
        for attr in ['cpu', 'gpu', 'object_store_memory']:
            if getattr(self.resource_limits, attr) is not None and getattr(self.exclude_resources, attr, 0) > 0:
                raise ValueError(f'resource_limits and exclude_resources cannot  both be set for {attr} resource.')