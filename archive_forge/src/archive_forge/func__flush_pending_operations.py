import logging
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from typing import Any, Dict, List, Optional, Union
from mlflow.entities import Metric, Param, RunTag
from mlflow.entities.dataset_input import DatasetInput
from mlflow.exceptions import MlflowException
from mlflow.tracking.client import MlflowClient
from mlflow.utils import _truncate_dict, chunk_list
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
def _flush_pending_operations(self, pending_operations):
    """
        Synchronously and sequentially flushes the specified list of pending run operations.

        NB: Operations are not parallelized on a per-run basis because MLflow's File Store, which
        is frequently used for local ML development, does not support threadsafe metadata logging
        within a given run.
        """
    if pending_operations.create_run:
        create_run_tags = pending_operations.create_run.tags
        num_additional_tags_to_include_during_creation = MAX_ENTITIES_PER_BATCH - len(create_run_tags)
        if num_additional_tags_to_include_during_creation > 0:
            create_run_tags.extend(pending_operations.tags_queue[:num_additional_tags_to_include_during_creation])
            pending_operations.tags_queue = pending_operations.tags_queue[num_additional_tags_to_include_during_creation:]
        new_run = self._client.create_run(experiment_id=pending_operations.create_run.experiment_id, start_time=pending_operations.create_run.start_time, tags={tag.key: tag.value for tag in create_run_tags})
        pending_operations.run_id = new_run.info.run_id
    run_id = pending_operations.run_id
    assert not isinstance(run_id, PendingRunId), 'Run ID cannot be pending for logging'
    operation_results = []
    param_batches_to_log = chunk_list(pending_operations.params_queue, chunk_size=MAX_PARAMS_TAGS_PER_BATCH)
    tag_batches_to_log = chunk_list(pending_operations.tags_queue, chunk_size=MAX_PARAMS_TAGS_PER_BATCH)
    for params_batch, tags_batch in zip_longest(param_batches_to_log, tag_batches_to_log, fillvalue=[]):
        metrics_batch_size = min(MAX_ENTITIES_PER_BATCH - len(params_batch) - len(tags_batch), MAX_METRICS_PER_BATCH)
        metrics_batch_size = max(metrics_batch_size, 0)
        metrics_batch = pending_operations.metrics_queue[:metrics_batch_size]
        pending_operations.metrics_queue = pending_operations.metrics_queue[metrics_batch_size:]
        operation_results.append(self._try_operation(self._client.log_batch, run_id=run_id, metrics=metrics_batch, params=params_batch, tags=tags_batch))
    for metrics_batch in chunk_list(pending_operations.metrics_queue, chunk_size=MAX_METRICS_PER_BATCH):
        operation_results.append(self._try_operation(self._client.log_batch, run_id=run_id, metrics=metrics_batch))
    for datasets_batch in chunk_list(pending_operations.datasets_queue, chunk_size=MAX_DATASETS_PER_BATCH):
        operation_results.append(self._try_operation(self._client.log_inputs, run_id=run_id, datasets=datasets_batch))
    if pending_operations.set_terminated:
        operation_results.append(self._try_operation(self._client.set_terminated, run_id=run_id, status=pending_operations.set_terminated.status, end_time=pending_operations.set_terminated.end_time))
    failures = [result for result in operation_results if isinstance(result, Exception)]
    if len(failures) > 0:
        raise MlflowException(message=f'Failed to perform one or more operations on the run with ID {run_id}. Failed operations: {failures}')