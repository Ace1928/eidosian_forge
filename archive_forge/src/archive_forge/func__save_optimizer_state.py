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
def _save_optimizer_state(self, path: pathlib.Path, optim: 'tf.keras.optimizers.Optimizer', optim_name: str) -> None:
    """Save the state variables of optim to path/optim_name_state.txt.

        Args:
            path: The path to the directory to save the state to.
            optim: The optimizer to save the state of.
            optim_name: The name of the optimizer.

        """
    state = optim.variables()
    serialized_tensors = [tf.io.serialize_tensor(tensor) for tensor in state]
    contents = tf.strings.join(serialized_tensors, separator='tensor: ')
    tf.io.write_file(str(path / f'{optim_name}_state.txt'), contents)