from gymnasium.spaces import Discrete, MultiDiscrete, Space
from typing import Union, Optional
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.utils.exploration.stochastic_sampling import StochasticSampling
from ray.rllib.utils.framework import TensorType
Initializes a SoftQ Exploration object.

        Args:
            action_space: The gym action space used by the environment.
            temperature: The temperature to divide model outputs by
                before creating the Categorical distribution to sample from.
            framework: One of None, "tf", "torch".
        