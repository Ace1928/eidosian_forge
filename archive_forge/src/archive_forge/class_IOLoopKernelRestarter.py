import time
import warnings
from typing import Any
from traitlets import Instance
from ..restarter import KernelRestarter
class IOLoopKernelRestarter(KernelRestarter):
    """Monitor and autorestart a kernel."""
    loop = Instance('tornado.ioloop.IOLoop')

    def _loop_default(self) -> Any:
        warnings.warn('IOLoopKernelRestarter.loop is deprecated in jupyter-client 5.2', DeprecationWarning, stacklevel=4)
        from tornado import ioloop
        return ioloop.IOLoop.current()
    _pcallback = None

    def start(self) -> None:
        """Start the polling of the kernel."""
        if self._pcallback is None:
            from tornado.ioloop import PeriodicCallback
            self._pcallback = PeriodicCallback(self.poll, 1000 * self.time_to_dead)
            self._pcallback.start()

    def stop(self) -> None:
        """Stop the kernel polling."""
        if self._pcallback is not None:
            self._pcallback.stop()
            self._pcallback = None