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
@PublicAPI
class NoPreprocessor(Preprocessor):

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        return self._obs_space.shape

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        self.check_shape(observation)
        return observation

    @override(Preprocessor)
    def write(self, observation: TensorType, array: np.ndarray, offset: int) -> None:
        array[offset:offset + self._size] = np.array(observation, copy=False).ravel()

    @property
    @override(Preprocessor)
    def observation_space(self) -> gym.Space:
        return self._obs_space