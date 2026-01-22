import collections
import copy
from collections.abc import MutableMapping
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gym
from gym import spaces
def _add_pixel_observation(self, wrapped_observation):
    if self._pixels_only:
        observation = collections.OrderedDict()
    elif self._observation_is_dict:
        observation = type(wrapped_observation)(wrapped_observation)
    else:
        observation = collections.OrderedDict()
        observation[STATE_KEY] = wrapped_observation
    pixel_observations = {pixel_key: self._render(**self._render_kwargs[pixel_key]) for pixel_key in self._pixel_keys}
    observation.update(pixel_observations)
    return observation