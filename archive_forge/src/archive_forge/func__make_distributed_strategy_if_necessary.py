import json
import logging
import pathlib
from typing import (
from ray.rllib.core.learner.learner import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.core.rl_module.tf.tf_rl_module import TfRLModule
from ray.rllib.policy.eager_tf_policy import _convert_to_tf
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.serialization import convert_numpy_to_python_primitives
from ray.rllib.utils.typing import Optimizer, Param, ParamDict, TensorType
def _make_distributed_strategy_if_necessary(self) -> 'tf.distribute.Strategy':
    """Create a distributed strategy for the learner.

        A stratgey is a tensorflow object that is used for distributing training and
        gradient computation across multiple devices. By default a no-op strategy is
        used that is not distributed.

        Returns:
            A strategy for the learner to use for distributed training.

        """
    if self._distributed:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    elif self._use_gpu:
        devices = tf.config.list_logical_devices('GPU')
        assert self._local_gpu_idx < len(devices), f'local_gpu_idx {self._local_gpu_idx} is not a valid GPU id or is not available.'
        local_gpu = [devices[self._local_gpu_idx].name]
        strategy = tf.distribute.MirroredStrategy(devices=local_gpu)
    else:
        strategy = tf.distribute.get_strategy()
    return strategy