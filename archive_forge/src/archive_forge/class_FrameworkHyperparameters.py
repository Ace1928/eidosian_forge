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
class FrameworkHyperparameters:
    """The framework specific hyper-parameters.

    Args:
        eager_tracing: Whether to trace the model in eager mode. This enables tf
            tracing mode by wrapping the loss function computation in a tf.function.
            This is useful for speeding up the training loop. However, it is not
            compatible with all tf operations. For example, tf.print is not supported
            in tf.function.
        torch_compile: Whether to use torch.compile() within the context of a given
            learner.
        what_to_compile: What to compile when using torch.compile(). Can be one of
            [TorchCompileWhatToCompile.complete_update,
            TorchCompileWhatToCompile.forward_train].
            If `complete_update`, the update step of the learner will be compiled. This
            includes the forward pass of the RLModule, the loss computation, and the
            optimizer step.
            If `forward_train`, only the forward methods (and therein the
            forward_train method) of the RLModule will be compiled.
            Either of the two may lead to different performance gains in different
            settings.
            `complete_update` promises the highest performance gains, but may not work
            in some settings. By compiling only forward_train, you may already get
            some speedups and avoid issues that arise from compiling the entire update.
        troch_compile_config: The TorchCompileConfig to use for compiling the RL
            Module in Torch.
    """
    eager_tracing: bool = True
    torch_compile: bool = False
    what_to_compile: str = TorchCompileWhatToCompile.FORWARD_TRAIN
    torch_compile_cfg: Optional['TorchCompileConfig'] = None

    def validate(self):
        if self.torch_compile:
            if self.what_to_compile not in [TorchCompileWhatToCompile.FORWARD_TRAIN, TorchCompileWhatToCompile.COMPLETE_UPDATE]:
                raise ValueError(f'what_to_compile must be one of [TorchCompileWhatToCompile.forward_train, TorchCompileWhatToCompile.complete_update] but is {self.what_to_compile}')
            if self.torch_compile_cfg is None:
                raise ValueError('torch_compile_cfg must be set when torch_compile is True.')