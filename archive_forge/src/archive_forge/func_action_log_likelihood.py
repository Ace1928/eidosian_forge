import gymnasium as gym
import numpy as np
import tree
from typing import Dict, Any, List
import logging
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import convert_ma_batch_to_sample_batch
from ray.rllib.utils.policy import compute_log_likelihoods_from_input_dict
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorType, SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
@Deprecated(old='OffPolicyEstimator.action_log_likelihood', new='ray.rllib.utils.policy.compute_log_likelihoods_from_input_dict', error=True)
def action_log_likelihood(self, batch: SampleBatchType) -> TensorType:
    log_likelihoods = compute_log_likelihoods_from_input_dict(self.policy, batch)
    return convert_to_numpy(log_likelihoods)