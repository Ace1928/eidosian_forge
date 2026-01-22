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
def _validate_ray_actor_options(self):
    if not isinstance(self.ray_actor_options, dict):
        raise TypeError(f'Got invalid type "{type(self.ray_actor_options)}" for ray_actor_options. Expected a dictionary.')
    allowed_ray_actor_options = {'accelerator_type', 'memory', 'num_cpus', 'num_gpus', 'object_store_memory', 'resources', 'runtime_env'}
    for option in self.ray_actor_options:
        if option not in allowed_ray_actor_options:
            raise ValueError(f"Specifying '{option}' in ray_actor_options is not allowed. Allowed options: {allowed_ray_actor_options}")
    ray_option_utils.validate_actor_options(self.ray_actor_options, in_options=True)
    if self.ray_actor_options.get('num_cpus') is None:
        self.ray_actor_options['num_cpus'] = 1