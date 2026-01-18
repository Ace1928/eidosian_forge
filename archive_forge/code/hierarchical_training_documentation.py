import argparse
from gymnasium.spaces import Discrete, Tuple
import logging
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.windy_maze_env import WindyMazeEnv, HierarchicalWindyMazeEnv
from ray.rllib.utils.test_utils import check_learning_achieved
Example of hierarchical training using the multi-agent API.

The example env is that of a "windy maze". The agent observes the current wind
direction and can either choose to stand still, or move in that direction.

You can try out the env directly with:

    $ python hierarchical_training.py --flat

A simple hierarchical formulation involves a high-level agent that issues goals
(i.e., go north / south / east / west), and a low-level agent that executes
these goals over a number of time-steps. This can be implemented as a
multi-agent environment with a top-level agent and low-level agents spawned
for each higher-level action. The lower level agent is rewarded for moving
in the right direction.

You can try this formulation with:

    $ python hierarchical_training.py  # gets ~100 rew after ~100k timesteps

Note that the hierarchical formulation actually converges slightly slower than
using --flat in this example.
