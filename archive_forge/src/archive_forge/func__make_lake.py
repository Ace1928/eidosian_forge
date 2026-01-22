import gymnasium as gym
import random
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override
def _make_lake(self):
    self.frozen_lake = gym.make('FrozenLake-v1', desc=self.MAPS[self.cur_level - 1], is_slippery=False)