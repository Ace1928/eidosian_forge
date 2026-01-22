import logging
import os
import tempfile
import warnings
from collections import defaultdict
from time import time
from traceback import format_exc
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple, Union
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator, cross_validate
from sklearn.model_selection._validation import _check_multimetric_scoring, _score
import ray.cloudpickle as cpickle
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.constants import TRAIN_DATASET_KEY
from ray.train.sklearn import SklearnCheckpoint
from ray.train.sklearn._sklearn_utils import _has_cpu_params, _set_cpu_params
from ray.train.trainer import BaseTrainer, GenDataset
from ray.util import PublicAPI
from ray.util.joblib import register_ray
def _validate_attributes(self):
    super()._validate_attributes()
    if self.label_column is not None and (not isinstance(self.label_column, str)):
        raise ValueError(f"`label_column` must be a string or None, got '{self.label_column}'")
    if self.params is not None and (not isinstance(self.params, dict)):
        raise ValueError(f"`params` must be a dict or None, got '{self.params}'")
    if not isinstance(self.return_train_score_cv, bool):
        raise ValueError(f"`return_train_score_cv` must be a boolean, got '{self.return_train_score_cv}'")
    if TRAIN_DATASET_KEY not in self.datasets:
        raise KeyError(f"'{TRAIN_DATASET_KEY}' key must be preset in `datasets`. Got {list(self.datasets.keys())}")
    if 'cv' in self.datasets:
        raise KeyError("'cv' is a reserved key. Please choose a different key for the dataset.")
    if not isinstance(self.parallelize_cv, bool) and self.parallelize_cv is not None:
        raise ValueError(f"`parallelize_cv` must be a bool or None, got '{self.parallelize_cv}'")
    scaling_config = self._validate_scaling_config(self.scaling_config)
    if self.cv and self.parallelize_cv and scaling_config.trainer_resources.get('GPU', 0):
        raise ValueError('`parallelize_cv` cannot be True if there are GPUs assigned to the trainer.')