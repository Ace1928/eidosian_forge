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
def _run_forward_pass(self, forward_fn_name: str, batch: NestedDict[Any], **kwargs) -> Dict[ModuleID, Mapping[ModuleID, Any]]:
    """This is a helper method that runs the forward pass for the given module.

        It uses forward_fn_name to get the forward pass method from the RLModule
        (e.g. forward_train vs. forward_exploration) and runs it on the given batch.

        Args:
            forward_fn_name: The name of the forward pass method to run.
            batch: The batch of multi-agent data (i.e. mapping from module ids to
                SampleBaches).
            **kwargs: Additional keyword arguments to pass to the forward function.

        Returns:
            The output of the forward pass the specified modules. The output is a
            mapping from module ID to the output of the forward pass.
        """
    module_ids = list(batch.shallow_keys())
    for module_id in module_ids:
        self._check_module_exists(module_id)
    outputs = {}
    for module_id in module_ids:
        rl_module = self._rl_modules[module_id]
        forward_fn = getattr(rl_module, forward_fn_name)
        outputs[module_id] = forward_fn(batch[module_id], **kwargs)
    return outputs