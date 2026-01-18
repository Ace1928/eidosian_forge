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
Build the TfLearner.

        This method is specific TfLearner. Before running super() it sets the correct
        distributing strategy with the right device, so that computational graph is
        placed on the correct device. After running super(), depending on eager_tracing
        flag it will decide whether to wrap the update function with tf.function or not.
        