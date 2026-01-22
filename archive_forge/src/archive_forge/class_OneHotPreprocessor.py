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
@DeveloperAPI
class OneHotPreprocessor(Preprocessor):
    """One-hot preprocessor for Discrete and MultiDiscrete spaces.

    .. testcode::
        :skipif: True

        self.transform(Discrete(3).sample())

    .. testoutput::

        np.array([0.0, 1.0, 0.0])

    .. testcode::
        :skipif: True

        self.transform(MultiDiscrete([2, 3]).sample())

    .. testoutput::

        np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        if isinstance(obs_space, gym.spaces.Discrete):
            return (self._obs_space.n,)
        else:
            return (np.sum(self._obs_space.nvec),)

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        self.check_shape(observation)
        return gym.spaces.utils.flatten(self._obs_space, observation).astype(np.float32)

    @override(Preprocessor)
    def write(self, observation: TensorType, array: np.ndarray, offset: int) -> None:
        array[offset:offset + self.size] = self.transform(observation)