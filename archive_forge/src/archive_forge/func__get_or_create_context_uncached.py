import functools
import threading
from contextlib import contextmanager
from .driver import driver, USE_NV_BINDING
def _get_or_create_context_uncached(self, devnum):
    """See also ``get_or_create_context(devnum)``.
        This version does not read the cache.
        """
    with self._lock:
        with driver.get_active_context() as ac:
            if not ac:
                return self._activate_context_for(0)
            else:
                ctx = self.gpus[ac.devnum].get_primary_context()
                if USE_NV_BINDING:
                    ctx_handle = int(ctx.handle)
                    ac_ctx_handle = int(ac.context_handle)
                else:
                    ctx_handle = ctx.handle.value
                    ac_ctx_handle = ac.context_handle.value
                if ctx_handle != ac_ctx_handle:
                    msg = 'Numba cannot operate on non-primary CUDA context {:x}'
                    raise RuntimeError(msg.format(ac_ctx_handle))
                ctx.prepare_for_use()
            return ctx