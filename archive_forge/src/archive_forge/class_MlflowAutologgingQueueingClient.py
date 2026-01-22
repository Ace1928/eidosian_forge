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
class MlflowAutologgingQueueingClient:
    """
    Efficiently implements a subset of MLflow Tracking's  `MlflowClient` and fluent APIs to provide
    automatic batching and async execution of run operations by way of queueing, as well as
    parameter / tag truncation for autologging use cases. Run operations defined by this client,
    such as `create_run` and `log_metrics`, enqueue data for future persistence to MLflow
    Tracking. Data is not persisted until the queue is flushed via the `flush()` method, which
    supports synchronous and asynchronous execution.

    MlflowAutologgingQueueingClient is not threadsafe; none of its APIs should be called
    concurrently.
    """

    def __init__(self, tracking_uri=None):
        self._client = MlflowClient(tracking_uri)
        self._pending_ops_by_run_id = {}

    def __enter__(self):
        """
        Enables `MlflowAutologgingQueueingClient` to be used as a context manager with
        synchronous flushing upon exit, removing the need to call `flush()` for use cases
        where logging completion can be waited upon synchronously.

        Run content is only flushed if the context exited without an exception.
        """
        return self

    def __exit__(self, exc_type, exc, traceback):
        """
        Enables `MlflowAutologgingQueueingClient` to be used as a context manager with
        synchronous flushing upon exit, removing the need to call `flush()` for use cases
        where logging completion can be waited upon synchronously.

        Run content is only flushed if the context exited without an exception.
        """
        if exc is None and exc_type is None and (traceback is None):
            self.flush(synchronous=True)
        else:
            _logger.debug('Skipping run content logging upon MlflowAutologgingQueueingClient context because an exception was raised within the context: %s', exc)

    def create_run(self, experiment_id: str, start_time: Optional[int]=None, tags: Optional[Dict[str, Any]]=None, run_name: Optional[str]=None) -> PendingRunId:
        """
        Enqueues a CreateRun operation with the specified attributes, returning a `PendingRunId`
        instance that can be used as input to other client logging APIs (e.g. `log_metrics`,
        `log_params`, ...).

        Returns:
            A `PendingRunId` that can be passed as the `run_id` parameter to other client
            logging APIs, such as `log_params` and `log_metrics`.
        """
        tags = tags or {}
        tags = _truncate_dict(tags, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_TAG_VAL_LENGTH)
        run_id = PendingRunId()
        self._get_pending_operations(run_id).enqueue(create_run=_PendingCreateRun(experiment_id=experiment_id, start_time=start_time, tags=[RunTag(key, str(value)) for key, value in tags.items()], run_name=run_name))
        return run_id

    def set_terminated(self, run_id: Union[str, PendingRunId], status: Optional[str]=None, end_time: Optional[int]=None) -> None:
        """
        Enqueues an UpdateRun operation with the specified `status` and `end_time` attributes
        for the specified `run_id`.
        """
        self._get_pending_operations(run_id).enqueue(set_terminated=_PendingSetTerminated(status=status, end_time=end_time))

    def log_params(self, run_id: Union[str, PendingRunId], params: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Parameters to be logged to the run specified by `run_id`.
        """
        params = _truncate_dict(params, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_PARAM_VAL_LENGTH)
        params_arr = [Param(key, str(value)) for key, value in params.items()]
        self._get_pending_operations(run_id).enqueue(params=params_arr)

    def log_inputs(self, run_id: Union[str, PendingRunId], datasets: Optional[List[DatasetInput]]) -> None:
        """
        Enqueues a collection of Dataset to be logged to the run specified by `run_id`.
        """
        if datasets is None or len(datasets) == 0:
            return
        self._get_pending_operations(run_id).enqueue(datasets=datasets)

    def log_metrics(self, run_id: Union[str, PendingRunId], metrics: Dict[str, float], step: Optional[int]=None) -> None:
        """
        Enqueues a collection of Metrics to be logged to the run specified by `run_id` at the
        step specified by `step`.
        """
        metrics = _truncate_dict(metrics, max_key_length=MAX_ENTITY_KEY_LENGTH)
        timestamp_ms = get_current_time_millis()
        metrics_arr = [Metric(key, value, timestamp_ms, step or 0) for key, value in metrics.items()]
        self._get_pending_operations(run_id).enqueue(metrics=metrics_arr)

    def set_tags(self, run_id: Union[str, PendingRunId], tags: Dict[str, Any]) -> None:
        """
        Enqueues a collection of Tags to be logged to the run specified by `run_id`.
        """
        tags = _truncate_dict(tags, max_key_length=MAX_ENTITY_KEY_LENGTH, max_value_length=MAX_TAG_VAL_LENGTH)
        tags_arr = [RunTag(key, str(value)) for key, value in tags.items()]
        self._get_pending_operations(run_id).enqueue(tags=tags_arr)

    def flush(self, synchronous=True):
        """
        Flushes all queued run operations, resulting in the creation or mutation of runs
        and run data.

        Args:
            synchronous: If `True`, run operations are performed synchronously, and a
                `RunOperations` result object is only returned once all operations
                are complete. If `False`, run operations are performed asynchronously,
                and an `RunOperations` object is returned that represents the ongoing
                run operations.

        Returns:
            A `RunOperations` instance representing the flushed operations. These operations
            are already complete if `synchronous` is `True`. If `synchronous` is `False`, these
            operations may still be inflight. Operation completion can be synchronously waited
            on via `RunOperations.await_completion()`.
        """
        logging_futures = []
        for pending_operations in self._pending_ops_by_run_id.values():
            future = _AUTOLOGGING_QUEUEING_CLIENT_THREAD_POOL.submit(self._flush_pending_operations, pending_operations=pending_operations)
            logging_futures.append(future)
        self._pending_ops_by_run_id = {}
        logging_operations = RunOperations(logging_futures)
        if synchronous:
            logging_operations.await_completion()
        return logging_operations

    def _get_pending_operations(self, run_id):
        """
        Returns:
            A `_PendingRunOperations` containing all pending operations for the
            specified `run_id`.
        """
        if run_id not in self._pending_ops_by_run_id:
            self._pending_ops_by_run_id[run_id] = _PendingRunOperations(run_id=run_id)
        return self._pending_ops_by_run_id[run_id]

    def _try_operation(self, fn, *args, **kwargs):
        """
        Attempt to evaluate the specified function, `fn`, on the specified `*args` and `**kwargs`,
        returning either the result of the function evaluation (if evaluation was successful) or
        the exception raised by the function evaluation (if evaluation was unsuccessful).
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return e

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