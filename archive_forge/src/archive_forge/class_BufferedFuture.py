import threading
from concurrent.futures import Future
from typing import Any, Callable, Generator, Generic, Optional, Tuple, Type, TypeVar
class BufferedFuture(AwaitableFuture):
    """A future whose async operation may be buffered until flush is called.

    Calling the flush method starts the asynchronous operation associated with
    this future, if it has not been started already. By default, calling
    result or exception will also call flush so that the async operation will
    start and we do not deadlock waiting for a result.
    """

    def flush(self):
        pass

    def result(self, timeout=None):
        self.flush()
        return super().result(timeout)

    def exception(self, timeout=None):
        self.flush()
        return super().exception(timeout)