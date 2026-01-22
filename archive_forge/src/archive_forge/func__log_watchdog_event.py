import json
import os
import shutil
import signal
import socket
from string import Template
import tempfile
import uuid
from typing import Any, Dict, Optional, Tuple
import torch.distributed.elastic.timer as timer
from torch.distributed.elastic import events
from torch.distributed.elastic.agent.server.api import (
from torch.distributed.elastic.events.api import EventMetadataValue
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import PContext, start_processes
from torch.distributed.elastic.utils import macros
from torch.distributed.elastic.utils.logging import get_logger
def _log_watchdog_event(self, name: str, request: Optional[timer.FileTimerRequest]) -> None:
    wg = self._worker_group
    spec = wg.spec
    md = {'watchdog_event': name}
    if request is not None:
        md['worker_pid'] = str(request.worker_pid)
        md['scope_id'] = request.scope_id
        md['expiration_time'] = str(request.expiration_time)
        md['signal'] = str(request.signal)
    md_str = json.dumps(md)
    state = 'RUNNING'
    metadata: Dict[str, EventMetadataValue] = {'run_id': spec.rdzv_handler.get_run_id(), 'global_rank': None, 'group_rank': wg.group_rank, 'worker_id': None, 'role': spec.role, 'hostname': self._get_fq_hostname(), 'state': state, 'total_run_time': self._total_execution_time, 'rdzv_backend': spec.rdzv_handler.get_backend(), 'raw_error': None, 'metadata': md_str, 'agent_restarts': spec.max_restarts - self._remaining_restarts}
    event = events.Event(name=name, source=events.EventSource.AGENT, metadata=metadata)
    events.record(event)