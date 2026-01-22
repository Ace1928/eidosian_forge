import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@PublicAPI(stability='stable')
class DeploymentSchema(BaseModel, allow_population_by_field_name=True):
    """
    Specifies options for one deployment within a Serve application. For each deployment
    this can optionally be included in `ServeApplicationSchema` to override deployment
    options specified in code.
    """
    name: str = Field(..., description='Globally-unique name identifying this deployment.')
    num_replicas: Optional[int] = Field(default=DEFAULT.VALUE, description='The number of processes that handle requests to this deployment. Uses a default if null.', gt=0)
    route_prefix: Union[str, None] = Field(default=DEFAULT.VALUE, description='[DEPRECATED] Please use route_prefix under ServeApplicationSchema instead.')
    max_concurrent_queries: int = Field(default=DEFAULT.VALUE, description='The max number of pending queries in a single replica. Uses a default if null.', gt=0)
    user_config: Optional[Dict] = Field(default=DEFAULT.VALUE, description="Config to pass into this deployment's reconfigure method. This can be updated dynamically without restarting replicas")
    autoscaling_config: Optional[Dict] = Field(default=DEFAULT.VALUE, description="Config specifying autoscaling parameters for the deployment's number of replicas. If null, the deployment won't autoscale its number of replicas; the number of replicas will be fixed at num_replicas.")
    graceful_shutdown_wait_loop_s: float = Field(default=DEFAULT.VALUE, description='Duration that deployment replicas will wait until there is no more work to be done before shutting down. Uses a default if null.', ge=0)
    graceful_shutdown_timeout_s: float = Field(default=DEFAULT.VALUE, description='Serve controller waits for this duration before forcefully killing the replica for shutdown. Uses a default if null.', ge=0)
    health_check_period_s: float = Field(default=DEFAULT.VALUE, description='Frequency at which the controller will health check replicas. Uses a default if null.', gt=0)
    health_check_timeout_s: float = Field(default=DEFAULT.VALUE, description="Timeout that the controller will wait for a response from the replica's health check before marking it unhealthy. Uses a default if null.", gt=0)
    ray_actor_options: RayActorOptionsSchema = Field(default=DEFAULT.VALUE, description='Options set for each replica actor.')
    placement_group_bundles: List[Dict[str, float]] = Field(default=DEFAULT.VALUE, description="Define a set of placement group bundles to be scheduled *for each replica* of this deployment. The replica actor will be scheduled in the first bundle provided, so the resources specified in `ray_actor_options` must be a subset of the first bundle's resources. All actors and tasks created by the replica actor will be scheduled in the placement group by default (`placement_group_capture_child_tasks` is set to True).")
    placement_group_strategy: str = Field(default=DEFAULT.VALUE, description='Strategy to use for the replica placement group specified via `placement_group_bundles`. Defaults to `PACK`.')
    max_replicas_per_node: int = Field(default=DEFAULT.VALUE, description='[EXPERIMENTAL] The max number of deployment replicas can run on a single node. Valid values are None (no limitation) or an integer in the range of [1, 100]. Defaults to no limitation.')
    logging_config: LoggingConfig = Field(default=DEFAULT.VALUE, description='Logging config for configuring serve deployment logs.')

    @root_validator
    def num_replicas_and_autoscaling_config_mutually_exclusive(cls, values):
        if values.get('num_replicas', None) not in [DEFAULT.VALUE, None] and values.get('autoscaling_config', None) not in [DEFAULT.VALUE, None]:
            raise ValueError('Manually setting num_replicas is not allowed when autoscaling_config is provided.')
        return values
    deployment_schema_route_prefix_format = validator('route_prefix', allow_reuse=True)(_route_prefix_format)

    def get_user_configured_option_names(self) -> Set[str]:
        """Get set of names for all user-configured options.

        Any field not set to DEFAULT.VALUE is considered a user-configured option.
        """
        return {field for field, value in self.dict().items() if value is not DEFAULT.VALUE}