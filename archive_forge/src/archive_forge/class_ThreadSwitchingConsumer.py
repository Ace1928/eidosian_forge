import logging
import threading
from grpc.framework.foundation import stream
class ThreadSwitchingConsumer(stream.Consumer):
    """A Consumer decorator that affords serialization and asynchrony."""

    def __init__(self, sink, pool):
        self._lock = threading.Lock()
        self._sink = sink
        self._pool = pool
        self._spinning = False
        self._values = []
        self._active = True

    def _spin(self, sink, value, terminate):
        while True:
            try:
                if value is _NO_VALUE:
                    sink.terminate()
                elif terminate:
                    sink.consume_and_terminate(value)
                else:
                    sink.consume(value)
            except Exception as e:
                _LOGGER.exception(e)
            with self._lock:
                if terminate:
                    self._spinning = False
                    return
                elif self._values:
                    value = self._values.pop(0)
                    terminate = not self._values and (not self._active)
                elif not self._active:
                    value = _NO_VALUE
                    terminate = True
                else:
                    self._spinning = False
                    return

    def consume(self, value):
        with self._lock:
            if self._active:
                if self._spinning:
                    self._values.append(value)
                else:
                    self._pool.submit(self._spin, self._sink, value, False)
                    self._spinning = True

    def terminate(self):
        with self._lock:
            if self._active:
                self._active = False
                if not self._spinning:
                    self._pool.submit(self._spin, self._sink, _NO_VALUE, True)
                    self._spinning = True

    def consume_and_terminate(self, value):
        with self._lock:
            if self._active:
                self._active = False
                if self._spinning:
                    self._values.append(value)
                else:
                    self._pool.submit(self._spin, self._sink, value, True)
                    self._spinning = True