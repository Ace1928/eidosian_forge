from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Union
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.images import rgb2gray, resize
@PublicAPI
def wrap_deepmind(env, dim=84, framestack=True, noframeskip=False):
    """Configure environment for DeepMind-style Atari.

    Note that we assume reward clipping is done outside the wrapper.

    Args:
        env: The env object to wrap.
        dim: Dimension to resize observations to (dim x dim).
        framestack: Whether to framestack observations.
    """
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if env.spec is not None and noframeskip is True:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, dim)
    if framestack is True:
        env = FrameStack(env, 4)
    return env