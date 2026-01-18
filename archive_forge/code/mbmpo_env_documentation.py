import numpy as np
import gymnasium as gym
Wrapper for the MuJoCo Hopper-v2 environment.

    Adds an additional `reward` method for some model-based RL algos (e.g.
    MB-MPO).
    