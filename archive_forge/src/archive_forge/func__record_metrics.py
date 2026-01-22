import abc
import functools
import json
import os
import signal
import socket
import time
import traceback
import warnings
from contextlib import closing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch.distributed.elastic.rendezvous as rdzv
import torch.distributed.elastic.utils.store as store_util
from torch.distributed import Store
from torch.distributed.elastic.events import Event, EventSource, record
from torch.distributed.elastic.metrics import prof, put_metric
from torch.distributed.elastic.multiprocessing import (
from torch.distributed.elastic.utils.logging import get_logger
def _record_metrics(self, group_results: RunResult):
    is_failed = group_results.is_failed()
    self._record_flakiness_metric(is_failed)
    spec = self._worker_group.spec
    restarts_happened = self._remaining_restarts != spec.max_restarts
    put_metric(f'workers.{spec.role}.run_total', 1)
    self._record_metric_with_condition('run_success_with_retries', not is_failed and restarts_happened)
    self._record_metric_with_condition('run_success_no_retries', not is_failed and (not restarts_happened))
    self._record_metric_with_condition('run_failed_with_retries', is_failed and restarts_happened)
    self._record_metric_with_condition('run_failed_no_retries', is_failed and (not restarts_happened))