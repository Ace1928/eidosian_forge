from collections import OrderedDict
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete
from gymnasium.wrappers import EnvCompatibility
import numpy as np
from recsim.document import AbstractDocumentSampler
from recsim.simulator import environment, recsim_gym
from recsim.user import AbstractUserModel, AbstractResponse
from typing import Callable, List, Optional, Type
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.gym import convert_old_gym_space_to_gymnasium_space
from ray.rllib.utils.spaces.space_utils import convert_element_to_space_type
class RecSimObservationBanditWrapper(gym.ObservationWrapper):
    """Fix RecSim environment's observation format

    RecSim's observations are keyed by document IDs, and nested under
    "doc" key.
    Our Bandits agent expects the observations to be flat 2D array
    and under "item" key.

    This environment wrapper converts obs into the right format.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = convert_old_gym_space_to_gymnasium_space(self.env.observation_space)
        num_items = len(obs_space['doc'])
        embedding_dim = next(iter(obs_space['doc'].values())).shape[-1]
        self.observation_space = Dict(OrderedDict([('item', gym.spaces.Box(low=-1.0, high=1.0, shape=(num_items, embedding_dim)))]))
        self._sampled_obs = self.observation_space.sample()
        self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space)

    def observation(self, obs):
        new_obs = OrderedDict()
        new_obs['item'] = np.vstack(list(obs['doc'].values()))
        new_obs = convert_element_to_space_type(new_obs, self._sampled_obs)
        return new_obs