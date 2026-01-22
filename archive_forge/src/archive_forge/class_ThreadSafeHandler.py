import os
import sys
import signal
import threading
import logging
from dataclasses import dataclass
from functools import partialmethod
from typing import Optional, Any, Dict, List
from .mp_utils import _CPU_CORES, _MAX_THREADS, _MAX_PROCS
class ThreadSafeHandler:

    def __init__(self, lock_mode: str='lock', handler: Optional[Any]=None):
        self.lock_mode = lock_mode
        self.handler = handler
        self.lock = threading.RLock() if self.lock_mode == 'rlock' else threading.Lock()

    def get(self, init_func: Any=None, *args, **kwargs):
        if self.lock_mode == 'rlock':
            self.lock.acquire()
        with self.lock:
            if not self.handler:
                self.handler = init_func(*args, **kwargs)
            return self.handler