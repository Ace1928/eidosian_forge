import abc
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Mapping, Any, TYPE_CHECKING, Optional, Type, Dict, Union
import gymnasium as gym
import tree
import ray
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.policy.policy import get_gym_space_from_struct_of_tensors
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.models.specs.checker import (
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.serialization import (
@classmethod
def _from_metadata_file(cls, metadata_path: Union[str, pathlib.Path]) -> 'RLModule':
    """Constructs a module from the metadata.

        Args:
            metadata_path: The path to the metadata json file for a module.

        Returns:
            The module.
        """
    metadata_path = pathlib.Path(metadata_path)
    if not metadata_path.exists():
        raise ValueError(f'While constructing the module from the metadata, the metadata file was not found at {str(metadata_path)}')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    module_spec_class = deserialize_type(metadata[RLMODULE_METADATA_SPEC_CLASS_KEY])
    module_spec = module_spec_class.from_dict(metadata[RLMODULE_METADATA_SPEC_KEY])
    module = module_spec.build()
    return module