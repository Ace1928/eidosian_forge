from gymnasium.spaces import Box, Discrete
import numpy as np
from typing import Optional, TYPE_CHECKING, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import Categorical, Deterministic
from ray.rllib.models.torch.torch_action_dist import (
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.exploration import Exploration
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.numpy import softmax, SMALL_NUMBER
from ray.rllib.utils.typing import TensorType
def _sample_new_noise(self, *, tf_sess=None):
    """Samples new noise and stores it in `self.noise`."""
    if self.framework == 'tf':
        tf_sess.run(self.tf_sample_new_noise_op)
    elif self.framework == 'tf2':
        self._tf_sample_new_noise_op()
    else:
        for i in range(len(self.noise)):
            self.noise[i] = torch.normal(mean=torch.zeros(self.noise[i].size()), std=self.stddev).to(self.device)