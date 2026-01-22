import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
Simple Env that yields a ts counter as observation (0-based).

    Actions have no effect.
    The episode length is always 15.
    Reward is always: current ts % 3.
    