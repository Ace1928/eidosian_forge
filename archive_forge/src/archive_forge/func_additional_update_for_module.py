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
@OverrideToImplementCustomLogic_CallToSuperRecommended
def additional_update_for_module(self, *, module_id: ModuleID, hps: LearnerHyperparameters, timestep: int, **kwargs) -> Dict[str, Any]:
    """Apply additional non-gradient based updates for a single module.

        See `additional_update` for more details.

        Args:
            module_id: The id of the module to update.
            hps: The LearnerHyperparameters specific to the given `module_id`.
            timestep: The current global timestep (to be used with schedulers).
            **kwargs: Keyword arguments to use for the additional update.

        Returns:
            A dictionary of results from the update
        """
    results = {}
    for optimizer_name, optimizer in self.get_optimizers_for_module(module_id):
        if optimizer in self._optimizer_lr_schedules:
            new_lr = self._optimizer_lr_schedules[optimizer].update(timestep=timestep)
            self._set_optimizer_lr(optimizer, lr=new_lr)
            stats_name = LEARNER_RESULTS_CURR_LR_KEY
            if optimizer_name != DEFAULT_OPTIMIZER:
                stats_name += '_' + optimizer_name
            results.update({stats_name: new_lr})
    return results