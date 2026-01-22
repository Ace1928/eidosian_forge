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
@classmethod
def __check_module_configs(cls, module_configs: Dict[ModuleID, Any]):
    """Checks the module configs for validity.

        The module_configs be a mapping from module_ids to SingleAgentRLModuleSpec
        objects.

        Args:
            module_configs: The module configs to check.

        Raises:
            ValueError: If the module configs are invalid.
        """
    for module_id, module_spec in module_configs.items():
        if not isinstance(module_spec, SingleAgentRLModuleSpec):
            raise ValueError(f'Module {module_id} is not a SingleAgentRLModuleSpec object.')