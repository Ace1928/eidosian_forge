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
def _validate_max_replicas_per_node(self) -> None:
    if self.max_replicas_per_node is None:
        return
    if not isinstance(self.max_replicas_per_node, int):
        raise TypeError(f"Get invalid type '{type(self.max_replicas_per_node)}' for max_replicas_per_node. Expected None or an integer in the range of [1, {MAX_REPLICAS_PER_NODE_MAX_VALUE}].")
    if self.max_replicas_per_node < 1 or self.max_replicas_per_node > MAX_REPLICAS_PER_NODE_MAX_VALUE:
        raise ValueError(f'Invalid max_replicas_per_node {self.max_replicas_per_node}. Valid values are None or an integer in the range of [1, {MAX_REPLICAS_PER_NODE_MAX_VALUE}].')