import argparse
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
Example of running a custom hand-coded policy alongside trainable policies.

This example has two policies:
    (1) a simple simple policy trained with PPO optimizer
    (2) a hand-coded policy that acts at random in the env (doesn't learn)

In the console output, you can see the PPO policy does much better than random:
Result for PPO_multi_cartpole_0:
  ...
  policy_reward_mean:
    learnable_policy: 185.23
    random: 21.255
  ...
