from functools import partial
import numpy as np
import gymnasium as gym
import logging
import tree
from typing import Dict, List, Type, Union
import ray
import ray.experimental.tf_utils
from ray.rllib.algorithms.sac.sac_tf_policy import (
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.policy.tf_mixins import TargetNetworkMixin
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.exploration.random import Random
from ray.rllib.utils.framework import get_variable, try_import_tf, try_import_tfp
from ray.rllib.utils.typing import (
def apply_gradients_fn(policy, optimizer, grads_and_vars):
    sac_results = sac_apply_gradients(policy, optimizer, grads_and_vars)
    if policy.config['lagrangian']:
        if policy.config['framework'] == 'tf2':
            policy._alpha_prime_optimizer.apply_gradients(policy._alpha_prime_grads_and_vars)
            return
        else:
            alpha_prime_apply_op = policy._alpha_prime_optimizer.apply_gradients(policy._alpha_prime_grads_and_vars, global_step=tf1.train.get_or_create_global_step())
            return tf.group([sac_results, alpha_prime_apply_op])
    return sac_results