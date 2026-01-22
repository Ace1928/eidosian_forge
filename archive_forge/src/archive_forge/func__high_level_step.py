import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import logging
import random
from ray.rllib.env import MultiAgentEnv
def _high_level_step(self, action):
    logger.debug('High level agent sets goal')
    self.current_goal = action
    self.steps_remaining_at_level = 25
    self.num_high_level_steps += 1
    self.low_level_agent_id = 'low_level_{}'.format(self.num_high_level_steps)
    obs = {self.low_level_agent_id: [self.cur_obs, self.current_goal]}
    rew = {self.low_level_agent_id: 0}
    done = truncated = {'__all__': False}
    return (obs, rew, done, truncated, {})