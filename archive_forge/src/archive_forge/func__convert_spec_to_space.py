import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.utils.annotations import PublicAPI
def _convert_spec_to_space(spec):
    if isinstance(spec, dict):
        return spaces.Dict({k: _convert_spec_to_space(v) for k, v in spec.items()})
    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, specs.BoundedArray):
        return spaces.Box(low=np.asscalar(spec.minimum), high=np.asscalar(spec.maximum), shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, specs.Array):
        return spaces.Box(low=-float('inf'), high=float('inf'), shape=spec.shape, dtype=spec.dtype)
    raise NotImplementedError('Could not convert `Array` spec of type {} to Gym space. Attempted to convert: {}'.format(type(spec), spec))