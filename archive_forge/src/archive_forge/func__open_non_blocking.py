import io
import json
import logging
import os
import select
import signal
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Set, Tuple
from torch.distributed.elastic.timer.api import TimerClient, TimerRequest
def _open_non_blocking(self) -> Optional[io.TextIOWrapper]:
    try:
        fd = os.open(self._file_path, os.O_WRONLY | os.O_NONBLOCK)
        return os.fdopen(fd, 'wt')
    except Exception:
        return None