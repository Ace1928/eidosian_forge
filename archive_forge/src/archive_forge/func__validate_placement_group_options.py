import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from ray import cloudpickle
from ray._private import ray_option_utils
from ray._private.protobuf_compat import message_to_dict
from ray._private.pydantic_compat import (
from ray._private.serialization import pickle_dumps
from ray._private.utils import resources_from_ray_options
from ray.serve._private.constants import (
from ray.serve._private.utils import DEFAULT, DeploymentOptionUpdateType
from ray.serve.config import AutoscalingConfig
from ray.serve.generated.serve_pb2 import AutoscalingConfig as AutoscalingConfigProto
from ray.serve.generated.serve_pb2 import DeploymentConfig as DeploymentConfigProto
from ray.serve.generated.serve_pb2 import DeploymentLanguage
from ray.serve.generated.serve_pb2 import EncodingType as EncodingTypeProto
from ray.serve.generated.serve_pb2 import LoggingConfig as LoggingConfigProto
from ray.serve.generated.serve_pb2 import ReplicaConfig as ReplicaConfigProto
from ray.util.placement_group import VALID_PLACEMENT_GROUP_STRATEGIES
def _validate_placement_group_options(self) -> None:
    if self.placement_group_strategy is not None and self.placement_group_strategy not in VALID_PLACEMENT_GROUP_STRATEGIES:
        raise ValueError(f"Invalid placement group strategy '{self.placement_group_strategy}'. Supported strategies are: {VALID_PLACEMENT_GROUP_STRATEGIES}.")
    if self.placement_group_strategy is not None and self.placement_group_bundles is None:
        raise ValueError('If `placement_group_strategy` is provided, `placement_group_bundles` must also be provided.')
    if self.placement_group_bundles is not None:
        if not isinstance(self.placement_group_bundles, list) or len(self.placement_group_bundles) == 0:
            raise ValueError('`placement_group_bundles` must be a non-empty list of resource dictionaries. For example: `[{"CPU": 1.0}, {"GPU": 1.0}]`.')
        for i, bundle in enumerate(self.placement_group_bundles):
            if not isinstance(bundle, dict) or not all((isinstance(k, str) for k in bundle.keys())) or (not all((isinstance(v, (int, float)) for v in bundle.values()))):
                raise ValueError('`placement_group_bundles` must be a non-empty list of resource dictionaries. For example: `[{"CPU": 1.0}, {"GPU": 1.0}]`.')
            if i == 0:
                bundle_cpu = bundle.get('CPU', 0)
                replica_actor_num_cpus = self.ray_actor_options.get('num_cpus', 0)
                if bundle_cpu < replica_actor_num_cpus:
                    raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `num_cpus` for the actor is {replica_actor_num_cpus} but the bundle only has {bundle_cpu} `CPU` specified.')
                bundle_gpu = bundle.get('GPU', 0)
                replica_actor_num_gpus = self.ray_actor_options.get('num_gpus', 0)
                if bundle_gpu < replica_actor_num_gpus:
                    raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `num_gpus` for the actor is {replica_actor_num_gpus} but the bundle only has {bundle_gpu} `GPU` specified.')
                replica_actor_resources = self.ray_actor_options.get('resources', {})
                for actor_resource, actor_value in replica_actor_resources.items():
                    bundle_value = bundle.get(actor_resource, 0)
                    if bundle_value < actor_value:
                        raise ValueError(f'When using `placement_group_bundles`, the replica actor will be placed in the first bundle, so the resource requirements for the actor must be a subset of the first bundle. `{actor_resource}` requirement for the actor is {actor_value} but the bundle only has {bundle_value} `{actor_resource}` specified.')