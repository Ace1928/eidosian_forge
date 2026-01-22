from gymnasium.spaces import Box, Dict, Discrete
import numpy as np
from ray.rllib.examples.env.random_env import RandomEnv
A randomly acting environment that publishes an action-mask each step.