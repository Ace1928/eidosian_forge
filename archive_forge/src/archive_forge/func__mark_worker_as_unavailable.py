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
def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
    assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)
    q = self._index_queues[worker_id]
    q.put(None)
    self._workers_status[worker_id] = False
    assert self._workers_done_event.is_set() == shutdown