import contextlib
import threading
class AsyncMetricsContext(threading.local):
    """A context for controlling metrics recording when async checkpoint is used.
  """

    def __init__(self):
        super().__init__()
        self._in_async_metrics_context = False

    def enter_async_metrics_context(self):
        self._in_async_metrics_context = True

    def exit_async_metrics_context(self):
        self._in_async_metrics_context = False

    def in_async_metrics_context(self):
        return self._in_async_metrics_context