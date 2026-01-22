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
def _construct_event(self, state: str, source: EventSource, worker: Optional[Worker]=None, raw_error: Optional[str]=None) -> Event:
    wg = self._worker_group
    spec = wg.spec
    md = {'group_world_size': wg.group_world_size, 'entry_point': spec.get_entrypoint_name()}
    if worker:
        md['local_rank'] = (worker.local_rank,)
        md['role_rank'] = (worker.role_rank,)
        md['role_world_size'] = (worker.role_world_size,)
        global_rank = worker.global_rank
        worker_id = str(worker.id)
    else:
        global_rank = None
        worker_id = None
    md_str = json.dumps(md)
    metadata = {'run_id': spec.rdzv_handler.get_run_id(), 'global_rank': global_rank, 'group_rank': wg.group_rank, 'worker_id': worker_id, 'role': spec.role, 'hostname': _get_fq_hostname(), 'state': state, 'total_run_time': self._total_execution_time, 'rdzv_backend': spec.rdzv_handler.get_backend(), 'raw_error': raw_error, 'metadata': md_str, 'agent_restarts': spec.max_restarts - self._remaining_restarts}
    return Event(f'torchelastic.worker.status.{state}', source=source, metadata=metadata)