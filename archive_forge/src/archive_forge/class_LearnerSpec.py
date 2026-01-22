import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
@dataclass
class LearnerSpec:
    """The spec for constructing Learner actors.

    Args:
        learner_class: The Learner class to use.
        module_spec: The underlying (MA)RLModule spec to completely define the module.
        module: Alternatively the RLModule instance can be passed in directly. This
            only works if the Learner is not an actor.
        backend_config: The backend config for properly distributing the RLModule.
        learner_hyperparameters: The extra config for the loss/additional update. This
            should be a subclass of LearnerHyperparameters. This is useful for passing
            in algorithm configs that contains the hyper-parameters for loss
            computation, change of training behaviors, etc. e.g lr, entropy_coeff.

    """
    learner_class: Type['Learner']
    module_spec: Union['SingleAgentRLModuleSpec', 'MultiAgentRLModuleSpec'] = None
    module: Optional['RLModule'] = None
    learner_group_scaling_config: LearnerGroupScalingConfig = field(default_factory=LearnerGroupScalingConfig)
    learner_hyperparameters: LearnerHyperparameters = field(default_factory=LearnerHyperparameters)
    framework_hyperparameters: FrameworkHyperparameters = field(default_factory=FrameworkHyperparameters)

    def get_params_dict(self) -> Dict[str, Any]:
        """Returns the parameters than be passed to the Learner constructor."""
        return {'module': self.module, 'module_spec': self.module_spec, 'learner_group_scaling_config': self.learner_group_scaling_config, 'learner_hyperparameters': self.learner_hyperparameters, 'framework_hyperparameters': self.framework_hyperparameters}

    def build(self) -> 'Learner':
        """Builds the Learner instance."""
        return self.learner_class(**self.get_params_dict())