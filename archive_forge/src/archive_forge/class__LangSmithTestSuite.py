from __future__ import annotations
import atexit
import concurrent.futures
import datetime
import functools
import inspect
import logging
import threading
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TypeVar, overload
import orjson
from typing_extensions import TypedDict
from langsmith import client as ls_client
from langsmith import env as ls_env
from langsmith import run_helpers as rh
from langsmith import schemas as ls_schemas
from langsmith import utils as ls_utils
class _LangSmithTestSuite:
    _instances: Optional[dict] = None
    _lock = threading.RLock()

    def __init__(self, client: Optional[ls_client.Client], experiment: ls_schemas.TracerSession, dataset: ls_schemas.Dataset):
        self.client = client or ls_client.Client()
        self._experiment = experiment
        self._dataset = dataset
        self._version: Optional[datetime.datetime] = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        atexit.register(_end_tests, self)

    @property
    def id(self):
        return self._dataset.id

    @property
    def experiment_id(self):
        return self._experiment.id

    @property
    def experiment(self):
        return self._experiment

    @classmethod
    def from_test(cls, client: Optional[ls_client.Client], func: Callable) -> _LangSmithTestSuite:
        client = client or ls_client.Client()
        test_suite_name = _get_test_suite_name(func)
        with cls._lock:
            if not cls._instances:
                cls._instances = {}
            if test_suite_name not in cls._instances:
                test_suite = _get_test_suite(client, test_suite_name)
                experiment = _start_experiment(client, test_suite)
                cls._instances[test_suite_name] = cls(client, experiment, test_suite)
        return cls._instances[test_suite_name]

    @property
    def name(self):
        return self._experiment.name

    def update_version(self, version: datetime.datetime) -> None:
        with self._lock:
            if self._version is None or version > self._version:
                self._version = version

    def get_version(self) -> Optional[datetime.datetime]:
        with self._lock:
            return self._version

    def submit_result(self, run_id: uuid.UUID, error: Optional[str]=None) -> None:
        self._executor.submit(self._submit_result, run_id, error)

    def _submit_result(self, run_id: uuid.UUID, error: Optional[str]=None) -> None:
        if error:
            self.client.create_feedback(run_id, key='pass', score=0, comment=f'Error: {repr(error)}')
        else:
            self.client.create_feedback(run_id, key='pass', score=1)

    def sync_example(self, example_id: uuid.UUID, inputs: dict, outputs: dict, metadata: dict) -> None:
        self._executor.submit(self._sync_example, example_id, inputs, outputs, metadata.copy())

    def _sync_example(self, example_id: uuid.UUID, inputs: dict, outputs: dict, metadata: dict) -> None:
        inputs_ = _serde_example_values(inputs)
        outputs_ = _serde_example_values(outputs)
        try:
            example = self.client.read_example(example_id=example_id)
            if inputs_ != example.inputs or outputs_ != example.outputs or str(example.dataset_id) != str(self.id):
                self.client.update_example(example_id=example.id, inputs=inputs_, outputs=outputs_, metadata=metadata, dataset_id=self.id)
        except ls_utils.LangSmithNotFoundError:
            example = self.client.create_example(example_id=example_id, inputs=inputs_, outputs=outputs_, dataset_id=self.id, metadata=metadata)
        if example.modified_at:
            self.update_version(example.modified_at)

    def wait(self):
        self._executor.shutdown(wait=True)