import threading
import time
from grpc.beta import implementations  # pylint: disable=unused-import
from grpc.beta import interfaces
from grpc.framework.foundation import callable_util
from grpc.framework.foundation import future
class _ChannelReadyFuture(future.Future):

    def __init__(self, channel):
        self._condition = threading.Condition()
        self._channel = channel
        self._matured = False
        self._cancelled = False
        self._done_callbacks = []

    def _block(self, timeout):
        until = None if timeout is None else time.time() + timeout
        with self._condition:
            while True:
                if self._cancelled:
                    raise future.CancelledError()
                elif self._matured:
                    return
                elif until is None:
                    self._condition.wait()
                else:
                    remaining = until - time.time()
                    if remaining < 0:
                        raise future.TimeoutError()
                    else:
                        self._condition.wait(timeout=remaining)

    def _update(self, connectivity):
        with self._condition:
            if not self._cancelled and connectivity is interfaces.ChannelConnectivity.READY:
                self._matured = True
                self._channel.unsubscribe(self._update)
                self._condition.notify_all()
                done_callbacks = tuple(self._done_callbacks)
                self._done_callbacks = None
            else:
                return
        for done_callback in done_callbacks:
            callable_util.call_logging_exceptions(done_callback, _DONE_CALLBACK_EXCEPTION_LOG_MESSAGE, self)

    def cancel(self):
        with self._condition:
            if not self._matured:
                self._cancelled = True
                self._channel.unsubscribe(self._update)
                self._condition.notify_all()
                done_callbacks = tuple(self._done_callbacks)
                self._done_callbacks = None
            else:
                return False
        for done_callback in done_callbacks:
            callable_util.call_logging_exceptions(done_callback, _DONE_CALLBACK_EXCEPTION_LOG_MESSAGE, self)
        return True

    def cancelled(self):
        with self._condition:
            return self._cancelled

    def running(self):
        with self._condition:
            return not self._cancelled and (not self._matured)

    def done(self):
        with self._condition:
            return self._cancelled or self._matured

    def result(self, timeout=None):
        self._block(timeout)
        return None

    def exception(self, timeout=None):
        self._block(timeout)
        return None

    def traceback(self, timeout=None):
        self._block(timeout)
        return None

    def add_done_callback(self, fn):
        with self._condition:
            if not self._cancelled and (not self._matured):
                self._done_callbacks.append(fn)
                return
        fn(self)

    def start(self):
        with self._condition:
            self._channel.subscribe(self._update, try_to_connect=True)

    def __del__(self):
        with self._condition:
            if not self._cancelled and (not self._matured):
                self._channel.unsubscribe(self._update)