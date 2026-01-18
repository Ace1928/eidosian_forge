import numpy as np
import gymnasium as gym
from typing import Dict
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict
from ray.rllib.utils.annotations import DeveloperAPI
def tiled_log_std(x):
    return tf.tile(tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])