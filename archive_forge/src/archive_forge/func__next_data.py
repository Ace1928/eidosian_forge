import functools
import itertools
import logging
import os
import queue
import threading
import warnings
from typing import Any, Callable, Iterable, TypeVar, Generic, List, Optional, Union
import multiprocessing as python_multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import torch.utils.data.graph_settings
from torch._utils import ExceptionWrapper
from . import (
from torch.utils.data.datapipes.datapipe import _IterDataPipeSerializationWrapper, _MapDataPipeSerializationWrapper
from . import _utils
def _next_data(self):
    while True:
        while self._rcvd_idx < self._send_idx:
            info = self._task_info[self._rcvd_idx]
            worker_id = info[0]
            if len(info) == 2 or self._workers_status[worker_id]:
                break
            del self._task_info[self._rcvd_idx]
            self._rcvd_idx += 1
        else:
            if not self._persistent_workers:
                self._shutdown_workers()
            raise StopIteration
        if len(self._task_info[self._rcvd_idx]) == 2:
            data = self._task_info.pop(self._rcvd_idx)[1]
            return self._process_data(data)
        assert not self._shutdown and self._tasks_outstanding > 0
        idx, data = self._get_data()
        self._tasks_outstanding -= 1
        if self._dataset_kind == _DatasetKind.Iterable:
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                if self._persistent_workers:
                    self._workers_status[data.worker_id] = False
                else:
                    self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue
        if idx != self._rcvd_idx:
            self._task_info[idx] += (data,)
        else:
            del self._task_info[idx]
            return self._process_data(data)