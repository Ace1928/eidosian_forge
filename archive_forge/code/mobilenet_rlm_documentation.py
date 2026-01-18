import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.examples.models.mobilenet_v2_encoder import (
from ray.rllib.core.models.configs import ActorCriticEncoderConfig
A PPORLModules with mobilenet v2 as an encoder.

    The idea behind this model is to demonstrate how we can bypass catalog to
    take full control over what models and action distribution are being built.
    In this example, we do this to modify an existing RLModule with a custom encoder.
    