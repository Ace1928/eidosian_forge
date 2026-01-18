import argparse
import gymnasium as gym
import os
import random
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
Simple example of how to restore only one of n agents from a trained
multi-agent Algorithm using Ray tune.

Control the number of agents and policies via --num-agents and --num-policies.
