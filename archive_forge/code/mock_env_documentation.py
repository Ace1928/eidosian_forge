import gymnasium as gym
import numpy as np
from typing import Optional
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.utils.annotations import override
A custom vector env that uses a single(!) CartPole sub-env.

    However, this env pretends to be a vectorized one to illustrate how one
    could create custom VectorEnvs w/o the need for actual vectorizations of
    sub-envs under the hood.
    