import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import ray
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.air.config import RunConfig, ScalingConfig
from ray.train import BackendConfig, Checkpoint, TrainingIterator
from ray.train._internal import session
from ray.train._internal.backend_executor import BackendExecutor, TrialInfo
from ray.train._internal.data_config import DataConfig
from ray.train._internal.session import _TrainingResult, get_session
from ray.train._internal.utils import construct_train_func
from ray.train.trainer import BaseTrainer, GenDataset
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
def _propagate_results(self, training_results: List[_TrainingResult]):
    first_worker_result = training_results[0]
    assert all((isinstance(result, _TrainingResult) for result in training_results))
    tune_session = get_session()
    worker_checkpoints = [result.checkpoint for result in training_results if result.checkpoint is not None]
    at_least_one_reported_checkpoint = len(worker_checkpoints) > 0
    if at_least_one_reported_checkpoint:
        tune_session.storage._update_checkpoint_index(first_worker_result.metrics)
    assert all((checkpoint.path == tune_session.storage.checkpoint_fs_path for checkpoint in worker_checkpoints))
    checkpoint = Checkpoint(filesystem=tune_session.storage.storage_filesystem, path=tune_session.storage.checkpoint_fs_path) if at_least_one_reported_checkpoint else None
    tracked_training_result = _TrainingResult(checkpoint=checkpoint, metrics=first_worker_result.metrics)
    logger.debug(f'Report (metrics, checkpoint) to the Tune session:\n  metrics={tracked_training_result.metrics}\n  checkpoint={tracked_training_result.checkpoint}')
    tune_session._report_training_result(tracked_training_result)