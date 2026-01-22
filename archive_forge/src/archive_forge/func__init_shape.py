from collections import OrderedDict
import logging
import numpy as np
import gymnasium as gym
from typing import Any, List
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.spaces.repeated import Repeated
from ray.rllib.utils.typing import TensorType
from ray.rllib.utils.images import resize
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type
@override(Preprocessor)
def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
    assert isinstance(self._obs_space, Repeated)
    child_space = obs_space.child_space
    self.child_preprocessor = get_preprocessor(child_space)(child_space, self._options)
    size = 1 + self.child_preprocessor.size * obs_space.max_len
    return (size,)