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
def _invoke_run(self, role: str=DEFAULT_ROLE) -> RunResult:
    spec = self._worker_group.spec
    role = spec.role
    log.info('[%s] starting workers for entrypoint: %s', role, spec.get_entrypoint_name())
    self._initialize_workers(self._worker_group)
    monitor_interval = spec.monitor_interval
    rdzv_handler = spec.rdzv_handler
    while True:
        assert self._worker_group.state != WorkerState.INIT
        time.sleep(monitor_interval)
        run_result = self._monitor_workers(self._worker_group)
        state = run_result.state
        self._worker_group.state = state
        put_metric(f'workers.{role}.remaining_restarts', self._remaining_restarts)
        put_metric(f'workers.{role}.{state.name.lower()}', 1)
        if state == WorkerState.SUCCEEDED:
            log.info('[%s] worker group successfully finished. Waiting %s seconds for other agents to finish.', role, self._exit_barrier_timeout)
            self._exit_barrier()
            return run_result
        elif state in {WorkerState.UNHEALTHY, WorkerState.FAILED}:
            if self._remaining_restarts > 0:
                log.info('[%s] Worker group %s. %s/%s attempts left; will restart worker group', role, state.name, self._remaining_restarts, spec.max_restarts)
                self._remaining_restarts -= 1
                self._restart_workers(self._worker_group)
            else:
                self._stop_workers(self._worker_group)
                self._worker_group.state = WorkerState.FAILED
                return run_result
        elif state == WorkerState.HEALTHY:
            num_nodes_waiting = rdzv_handler.num_nodes_waiting()
            group_rank = self._worker_group.group_rank
            if num_nodes_waiting > 0:
                log.info('[%s] Detected %s new nodes from group_rank=%s; will restart worker group', role, num_nodes_waiting, group_rank)
                self._restart_workers(self._worker_group)
        else:
            raise Exception(f'[{role}] Worker group in {state.name} state')