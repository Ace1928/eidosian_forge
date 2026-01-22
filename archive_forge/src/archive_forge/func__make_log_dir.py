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
def _make_log_dir(self, log_dir: Optional[str], rdzv_run_id: str):
    base_log_dir = log_dir or tempfile.mkdtemp(prefix='torchelastic_')
    os.makedirs(base_log_dir, exist_ok=True)
    dir = tempfile.mkdtemp(prefix=f'{rdzv_run_id}_', dir=base_log_dir)
    log.info('log directory set to: %s', dir)
    return dir