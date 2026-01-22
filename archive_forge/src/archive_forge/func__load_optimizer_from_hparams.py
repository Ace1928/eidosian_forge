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
def _load_optimizer_from_hparams(self, path: pathlib.Path, optim_name: str) -> 'tf.keras.optimizers.Optimizer':
    """Load an optimizer from the hyperparameters saved at path/optim_name_hparams.json.

        Args:
            path: The path to the directory to load the hyperparameters from.
            optim_name: The name of the optimizer.

        Returns:
            The optimizer loaded from the hyperparameters.

        """
    with open(path / f'{optim_name}_hparams.json', 'r') as f:
        state = json.load(f)
    return tf.keras.optimizers.deserialize(state)