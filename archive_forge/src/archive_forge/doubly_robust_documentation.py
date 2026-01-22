import logging
import numpy as np
import math
import pandas as pd
from typing import Dict, Any, Optional, List
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.offline.estimators.fqe_torch_model import FQETorchModel
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.offline_evaluation_utils import (
Estimates the policy value using the Doubly Robust estimator.

        The doubly robust estimator uses normalization of importance sampling weights
        (aka. propensity ratios) to the average of the importance weights across the
        entire dataset. This is done to reduce the variance of the estimate (similar to
        weighted importance sampling). You can disable this by setting
        `normalize_weights=False` in the constructor.

        Note: This estimate works for only discrete action spaces for now.

        Args:
            dataset: Dataset to compute the estimate on. Each record in dataset should
                include the following columns: `obs`, `actions`, `action_prob` and
                `rewards`. The `obs` on each row shoud be a vector of D dimensions.
            n_parallelism: Number of parallelism to use for the computation.

        Returns:
            A dict with the following keys:
                v_target: The estimated value of the target policy.
                v_behavior: The estimated value of the behavior policy.
                v_gain: The estimated gain of the target policy over the behavior
                    policy.
                v_std: The standard deviation of the estimated value of the target.
        