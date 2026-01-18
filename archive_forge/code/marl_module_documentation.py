from dataclasses import dataclass, field
import pathlib
import pprint
from typing import (
from ray.util.annotations import PublicAPI
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.serialization import serialize_type, deserialize_type
from ray.rllib.utils.typing import T
Updates this spec with the other spec.

        Traverses this MultiAgentRLModuleSpec's module_specs and updates them with
        the module specs from the other MultiAgentRLModuleSpec.

        Args:
            other: The other spec to update this spec with.
            overwrite: Whether to overwrite the existing module specs if they already
                exist. If False, they will be updated only.
        