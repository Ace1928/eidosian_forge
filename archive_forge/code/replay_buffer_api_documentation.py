import argparse
import ray
from ray import air, tune
from ray.rllib.algorithms.r2d2 import R2D2Config
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.replay_buffers.replay_buffer import StorageUnit
Simple example of how to modify replay buffer behaviour.

We modify R2D2 to utilize prioritized replay but supplying it with the
PrioritizedMultiAgentReplayBuffer instead of the standard MultiAgentReplayBuffer.
This is possible because R2D2 uses the DQN training iteration function,
which includes and a priority update, given that a fitting buffer is provided.
