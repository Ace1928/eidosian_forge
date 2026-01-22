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
class RecSimObservationSpaceWrapper(gym.ObservationWrapper):
    """Fix RecSim environment's observation space

    In RecSim's observation spaces, the "doc" field is a dictionary keyed by
    document IDs. Those IDs are changing every step, thus generating a
    different observation space in each time. This causes issues for RLlib
    because it expects the observation space to remain the same across steps.

    This environment wrapper fixes that by reindexing the documents by their
    positions in the list.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_space = convert_old_gym_space_to_gymnasium_space(self.env.observation_space)
        doc_space = Dict(OrderedDict([(str(k), doc) for k, (_, doc) in enumerate(obs_space['doc'].spaces.items())]))
        self.observation_space = Dict(OrderedDict([('user', obs_space['user']), ('doc', doc_space), ('response', obs_space['response'])]))
        self._sampled_obs = self.observation_space.sample()
        self.action_space = convert_old_gym_space_to_gymnasium_space(self.env.action_space)

    def observation(self, obs):
        new_obs = OrderedDict()
        new_obs['user'] = obs['user']
        new_obs['doc'] = {str(k): v for k, (_, v) in enumerate(obs['doc'].items())}
        new_obs['response'] = obs['response']
        new_obs = convert_element_to_space_type(new_obs, self._sampled_obs)
        return new_obs