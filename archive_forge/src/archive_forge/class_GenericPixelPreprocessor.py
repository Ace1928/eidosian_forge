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
class GenericPixelPreprocessor(Preprocessor):
    """Generic image preprocessor.

    Note: for Atari games, use config {"preprocessor_pref": "deepmind"}
    instead for deepmind-style Atari preprocessing.
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space: gym.Space, options: dict) -> List[int]:
        self._grayscale = options.get('grayscale')
        self._zero_mean = options.get('zero_mean')
        self._dim = options.get('dim')
        if self._grayscale:
            shape = (self._dim, self._dim, 1)
        else:
            shape = (self._dim, self._dim, 3)
        return shape

    @override(Preprocessor)
    def transform(self, observation: TensorType) -> np.ndarray:
        """Downsamples images from (210, 160, 3) by the configured factor."""
        self.check_shape(observation)
        scaled = observation[25:-25, :, :]
        if self._dim < 84:
            scaled = resize(scaled, height=84, width=84)
        scaled = resize(scaled, height=self._dim, width=self._dim)
        if self._grayscale:
            scaled = scaled.mean(2)
            scaled = scaled.astype(np.float32)
            scaled = np.reshape(scaled, [self._dim, self._dim, 1])
        if self._zero_mean:
            scaled = (scaled - 128) / 128
        else:
            scaled *= 1.0 / 255.0
        return scaled