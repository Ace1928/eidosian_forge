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
def _shutdown_workers(self):
    if _utils is None or _utils.python_exit_status is True or _utils.python_exit_status is None:
        return
    if not self._shutdown:
        self._shutdown = True
        try:
            if hasattr(self, '_pin_memory_thread'):
                self._pin_memory_thread_done_event.set()
                self._worker_result_queue.put((None, None))
                self._pin_memory_thread.join()
                self._worker_result_queue.cancel_join_thread()
                self._worker_result_queue.close()
            self._workers_done_event.set()
            for worker_id in range(len(self._workers)):
                if self._persistent_workers or self._workers_status[worker_id]:
                    self._mark_worker_as_unavailable(worker_id, shutdown=True)
            for w in self._workers:
                w.join(timeout=_utils.MP_STATUS_CHECK_INTERVAL)
            for q in self._index_queues:
                q.cancel_join_thread()
                q.close()
        finally:
            if self._worker_pids_set:
                _utils.signal_handling._remove_worker_pids(id(self))
                self._worker_pids_set = False
            for w in self._workers:
                if w.is_alive():
                    w.terminate()