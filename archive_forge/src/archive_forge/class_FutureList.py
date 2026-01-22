import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
class FutureList(BufferedFuture):
    """A Future that waits for a list of other Futures."""

    def __init__(self, futures):
        super().__init__()
        if not len(futures):
            self.set_result([])
            return
        self._results = [None] * len(futures)
        self._outstanding = len(futures)
        self._lock = threading.Lock()
        self._buffer = BufferGroup()
        for i, f in enumerate(futures):
            self._buffer.add(f)
            f.add_done_callback(lambda f, idx=i: self._handle_result(f, idx))

    def _handle_result(self, future, index):
        if self.done():
            return
        error = future.exception()
        if error is not None:
            self.try_set_exception(error)
            return
        result = future.result()
        with self._lock:
            self._results[index] = result
            self._outstanding -= 1
            if not self._outstanding:
                self.try_set_result(self._results)

    def flush(self):
        self._buffer.flush()