import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tree  # pip install dm_tree
from typing import Dict, List, Optional
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.spaces.simplex import Simplex
from ray.rllib.utils.typing import ModelConfigDict, TensorType, TensorStructType
Concat model outs if they are original tuple observations.