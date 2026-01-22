import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def _activate_context_for(self, devnum):
    with self._lock:
        gpu = self.gpus[devnum]
        newctx = gpu.get_primary_context()
        cached_ctx = self._get_attached_context()
        if cached_ctx is not None and cached_ctx is not newctx:
            raise RuntimeError('Cannot switch CUDA-context.')
        newctx.push()
        return newctx