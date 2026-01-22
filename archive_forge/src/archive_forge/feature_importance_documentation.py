import copy
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
import ray
from ray.data import Dataset
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch, convert_ma_batch_to_sample_batch
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.offline.offline_evaluator import OfflineEvaluator
Estimate the feature importance of the policy given a dataset.

        For each feature in the dataset, the importance is computed by applying
        perturbations to each feature and computing the difference between the
        perturbed prediction and the reference prediction. The importance
        computation for each feature and each perturbation is repeated `self.repeat`
        times. If dataset is large the user can initialize the estimator with a
        `limit_fraction` to limit the dataset to a fraction of the original dataset.

        The dataset should include a column named `obs` where each row is a vector of D
        dimensions. The importance is computed for each dimension of the vector.

        Note (Implementation detail): The computation across features are distributed
        with ray workers since each feature is independent of each other.

        Args:
            dataset: the dataset to use for feature importance.
            n_parallelism: number of parallel workers to use for feature importance.

        Returns:
            A dict mapping each feature index string to its importance.
        