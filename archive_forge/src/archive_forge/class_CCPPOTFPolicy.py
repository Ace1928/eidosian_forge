import argparse
import numpy as np
from gymnasium.spaces import Discrete
import os
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import (
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.examples.models.centralized_critic_models import (
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.utils.tf_utils import explained_variance, make_tf_callable
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
class CCPPOTFPolicy(CentralizedValueMixin, base):

    def __init__(self, observation_space, action_space, config):
        base.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(base)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(base)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralized_critic_postprocessing(self, sample_batch, other_agent_batches, episode)

    @override(base)
    def stats_fn(self, train_batch: SampleBatch):
        stats = super().stats_fn(train_batch)
        stats.update(central_vf_stats(self, train_batch))
        return stats