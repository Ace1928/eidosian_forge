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
def training_loop(self) -> None:
    register_ray()
    self.estimator.set_params(**self.params)
    datasets = self._get_datasets()
    X_train, y_train = datasets.pop(TRAIN_DATASET_KEY)
    groups = None
    if 'cv_groups' in X_train.columns:
        groups = X_train['cv_groups']
        X_train = X_train.drop('cv_groups', axis=1)
    scaling_config = self._validate_scaling_config(self.scaling_config)
    num_workers = scaling_config.num_workers or 0
    assert num_workers == 0
    trainer_resources = scaling_config.trainer_resources or {'CPU': 1}
    has_gpus = bool(trainer_resources.get('GPU', 0))
    num_cpus = int(trainer_resources.get('CPU', 1))
    os.environ['OMP_NUM_THREADS'] = str(num_cpus)
    os.environ['MKL_NUM_THREADS'] = str(num_cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_cpus)
    os.environ['BLIS_NUM_THREADS'] = str(num_cpus)
    parallelize_cv = self._get_cv_parallelism(has_gpus)
    if self.set_estimator_cpus:
        num_estimator_cpus = 1 if parallelize_cv else num_cpus
        _set_cpu_params(self.estimator, num_estimator_cpus)
    with parallel_backend('ray', n_jobs=num_cpus):
        start_time = time()
        self.estimator.fit(X_train, y_train, **self.fit_params)
        fit_time = time() - start_time
        if self.label_column:
            validation_set_scores = self._score_on_validation_sets(self.estimator, datasets)
            cv_scores = self._score_cv(self.estimator, X_train, y_train, groups, n_jobs=1 if not parallelize_cv else num_cpus)
        else:
            validation_set_scores = {}
            cv_scores = {}
    results = {**validation_set_scores, **cv_scores, 'fit_time': fit_time, 'done': True}
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        checkpoint_file = os.path.join(temp_checkpoint_dir, SklearnCheckpoint.MODEL_FILENAME)
        with open(checkpoint_file, 'wb') as f:
            cpickle.dump(self.estimator, f)
        train.report(results, checkpoint=Checkpoint.from_directory(temp_checkpoint_dir))