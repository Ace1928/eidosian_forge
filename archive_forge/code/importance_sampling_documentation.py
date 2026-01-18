from typing import Dict, List, Any
import math
from ray.data import Dataset
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
from ray.rllib.offline.offline_evaluation_utils import (
from ray.rllib.offline.estimators.off_policy_estimator import OffPolicyEstimator
from ray.rllib.policy.sample_batch import SampleBatch
Computes the Importance sampling estimate on the given dataset.

        Note: This estimate works for both continuous and discrete action spaces.

        Args:
            dataset: Dataset to compute the estimate on. Each record in dataset should
                include the following columns: `obs`, `actions`, `action_prob` and
                `rewards`. The `obs` on each row shoud be a vector of D dimensions.
            n_parallelism: The number of parallel workers to use.

        Returns:
            A dictionary containing the following keys:
                v_target: The estimated value of the target policy.
                v_behavior: The estimated value of the behavior policy.
                v_gain_mean: The mean of the gain of the target policy over the
                    behavior policy.
                v_gain_ste: The standard error of the gain of the target policy over
                    the behavior policy.
        