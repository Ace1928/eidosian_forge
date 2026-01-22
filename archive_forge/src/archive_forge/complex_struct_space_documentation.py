import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.env.simple_rpg import SimpleRPG
from ray.rllib.examples.models.simple_rpg_model import (
Example of using variable-length Repeated / struct observation spaces.

This example shows:
  - using a custom environment with Repeated / struct observations
  - using a custom model to view the batched list observations

For PyTorch / TF eager mode, use the `--framework=[torch|tf2]` flag.
