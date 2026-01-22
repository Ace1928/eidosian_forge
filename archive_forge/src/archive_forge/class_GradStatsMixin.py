import logging
from typing import Dict, List
import numpy as np
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.eager_tf_policy import EagerTFPolicy
from ray.rllib.policy.eager_tf_policy_v2 import EagerTFPolicyV2
from ray.rllib.policy.policy import Policy, PolicyState
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.framework import get_variable, try_import_tf
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.utils.typing import (
@DeveloperAPI
class GradStatsMixin:

    def __init__(self):
        pass

    def grad_stats_fn(self, train_batch: SampleBatch, grads: ModelGradients) -> Dict[str, TensorType]:
        if self.config.get('_tf_policy_handles_more_than_one_loss'):
            grad_gnorm = [tf.linalg.global_norm(g) for g in grads]
        else:
            grad_gnorm = tf.linalg.global_norm(grads)
        return {'grad_gnorm': grad_gnorm}