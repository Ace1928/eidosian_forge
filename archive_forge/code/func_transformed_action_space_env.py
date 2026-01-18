import gymnasium as gym
from typing import Type
def transformed_action_space_env(config):
    if isinstance(env_name_or_creator, str):
        inner_env = gym.make(env_name_or_creator)
    else:
        inner_env = env_name_or_creator(config)
    _low = config.pop('low', -1.0)
    _high = config.pop('high', 1.0)
    env = ActionTransform(inner_env, _low, _high)
    return env