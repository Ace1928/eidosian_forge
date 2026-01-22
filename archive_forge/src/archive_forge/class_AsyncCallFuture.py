import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
class AsyncCallFuture:
    """Encapsulates the future value of a retriable async gRPC request.

    Abstracts over the set of futures returned by a set of gRPC calls
    comprising a single logical gRPC request with retries.  Communicates
    to the caller the result or exception resulting from the request.

    Args:
      completion_event: The constructor should provide a `threding.Event` which
        will be used to communicate when the set of gRPC requests is complete.
    """

    def __init__(self, completion_event):
        self._active_grpc_future = None
        self._active_grpc_future_lock = threading.Lock()
        self._completion_event = completion_event

    def _set_active_future(self, grpc_future):
        if grpc_future is None:
            raise RuntimeError('_set_active_future invoked with grpc_future=None.')
        with self._active_grpc_future_lock:
            self._active_grpc_future = grpc_future

    def result(self, timeout):
        """Analogous to `grpc.Future.result`. Returns the value or exception.

        This method will wait until the full set of gRPC requests is complete
        and then act as `grpc.Future.result` for the single gRPC invocation
        corresponding to the first successful call or final failure, as
        appropriate.

        Args:
          timeout: How long to wait in seconds before giving up and raising.

        Returns:
          The result of the future corresponding to the single gRPC
          corresponding to the successful call.

        Raises:
          * `grpc.FutureTimeoutError` if timeout seconds elapse before the gRPC
          calls could complete, including waits and retries.
          * The exception corresponding to the last non-retryable gRPC request
          in the case that a successful gRPC request was not made.
        """
        if not self._completion_event.wait(timeout):
            raise grpc.FutureTimeoutError(f'AsyncCallFuture timed out after {timeout} seconds')
        with self._active_grpc_future_lock:
            if self._active_grpc_future is None:
                raise RuntimeError('AsyncFuture never had an active future set')
            return self._active_grpc_future.result()