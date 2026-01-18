import argparse
import gymnasium as gym
import os
import ray
from ray import air, tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import get_trainable_cls
Create CLI parser and return parsed arguments