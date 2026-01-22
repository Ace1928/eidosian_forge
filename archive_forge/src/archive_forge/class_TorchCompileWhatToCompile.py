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
class TorchCompileWhatToCompile(str, Enum):
    """Enumerates schemes of what parts of the TorchLearner can be compiled.

    This can be either the entire update step of the learner or only the forward
    methods (and therein the forward_train method) of the RLModule.

    .. note::
        - torch.compiled code can become slow on graph breaks or even raise
            errors on unsupported operations. Empirically, compiling
            `forward_train` should introduce little graph breaks, raise no
            errors but result in a speedup comparable to compiling the
            complete update.
        - Using `complete_update` is experimental and may result in errors.
    """
    COMPLETE_UPDATE = 'complete_update'
    FORWARD_TRAIN = 'forward_train'