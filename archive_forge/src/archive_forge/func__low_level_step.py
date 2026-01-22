import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Tuple
import logging
import random
from ray.rllib.env import MultiAgentEnv
def _low_level_step(self, action):
    logger.debug('Low level agent step {}'.format(action))
    self.steps_remaining_at_level -= 1
    cur_pos = tuple(self.cur_obs[0])
    goal_pos = self.flat_env._get_new_pos(cur_pos, self.current_goal)
    f_obs, f_rew, f_terminated, f_truncated, info = self.flat_env.step(action)
    new_pos = tuple(f_obs[0])
    self.cur_obs = f_obs
    obs = {self.low_level_agent_id: [f_obs, self.current_goal]}
    if new_pos != cur_pos:
        if new_pos == goal_pos:
            rew = {self.low_level_agent_id: 1}
        else:
            rew = {self.low_level_agent_id: -1}
    else:
        rew = {self.low_level_agent_id: 0}
    terminated = {'__all__': False}
    truncated = {'__all__': False}
    if f_terminated or f_truncated:
        terminated['__all__'] = f_terminated
        truncated['__all__'] = f_truncated
        logger.debug('high level final reward {}'.format(f_rew))
        rew['high_level_agent'] = f_rew
        obs['high_level_agent'] = f_obs
    elif self.steps_remaining_at_level == 0:
        terminated[self.low_level_agent_id] = True
        truncated[self.low_level_agent_id] = False
        rew['high_level_agent'] = 0
        obs['high_level_agent'] = f_obs
    return (obs, rew, terminated, truncated, {self.low_level_agent_id: info})