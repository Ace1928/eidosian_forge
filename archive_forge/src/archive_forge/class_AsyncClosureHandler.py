import os
import threading
from queue import Empty as EmptyQueue, Queue
from torch._lazy.device_context import get_device_context
class AsyncClosureHandler(ClosureHandler):
    """Handler for Asynchronous Step Closures
    Args:
        max_queue_size: The maximum length of the closure queue after which
        the training loop will block until closures are evaluated.
        By default, a reasonable limit of a maximum of 100 on the queue.
        This value can be set using the `XLA_MAX_ASYNC_QUEUE` environment
        variable.
    """

    def __init__(self, max_queue_size=100):
        super().__init__()
        self._closure_queue: Queue = Queue(int(os.environ.get('LTC_MAX_ASYNC_QUEUE', max_queue_size)))
        self._closure_exception: Queue = Queue()
        self._closure_lock = threading.Lock()
        self._closure_event_loop_finished = threading.Event()
        self._closure_event_loop = None

    def start_event_loop(self):
        """Start closure event loop if not started"""
        if self._closure_event_loop is None:

            def event_loop():
                while True:
                    try:
                        closure = self._closure_queue.get(block=True, timeout=3)
                        closure()
                        self._closure_queue.task_done()
                    except EmptyQueue:
                        with self._closure_lock:
                            if self._closure_queue.empty():
                                self._closure_event_loop_finished.set()
                                return
                    except Exception as e:
                        self._closure_exception.put(e)
                        return
            self._closure_event_loop = threading.Thread(target=event_loop)
            self._closure_event_loop.start()

    def run(self, closure):
        with self._closure_lock:
            self._closure_queue.put(closure, block=True)
            if self._closure_event_loop is None or not self._closure_event_loop.is_alive():
                try:
                    e = self._closure_exception.get(block=False)
                    raise RuntimeError('Cannot run asynchronous closure due to previously raised exception') from e
                except EmptyQueue:
                    self._closure_event_loop = None
                    self.start_event_loop()