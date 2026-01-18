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
Retrieve the sklearn estimator stored in this checkpoint.