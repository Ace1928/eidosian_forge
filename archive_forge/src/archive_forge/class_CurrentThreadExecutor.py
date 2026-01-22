import queue
import sys
import threading
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union
class CurrentThreadExecutor(Executor):
    """
    An Executor that actually runs code in the thread it is instantiated in.
    Passed to other threads running async code, so they can run sync code in
    the thread they came from.
    """

    def __init__(self) -> None:
        self._work_thread = threading.current_thread()
        self._work_queue: queue.Queue[Union[_WorkItem, 'Future[Any]']] = queue.Queue()
        self._broken = False

    def run_until_future(self, future: 'Future[Any]') -> None:
        """
        Runs the code in the work queue until a result is available from the future.
        Should be run from the thread the executor is initialised in.
        """
        if threading.current_thread() != self._work_thread:
            raise RuntimeError('You cannot run CurrentThreadExecutor from a different thread')
        future.add_done_callback(self._work_queue.put)
        try:
            while True:
                work_item = self._work_queue.get()
                if work_item is future:
                    return
                assert isinstance(work_item, _WorkItem)
                work_item.run()
                del work_item
        finally:
            self._broken = True

    def _submit(self, fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> 'Future[_R]':
        if threading.current_thread() == self._work_thread:
            raise RuntimeError('You cannot submit onto CurrentThreadExecutor from its own thread')
        if self._broken:
            raise RuntimeError('CurrentThreadExecutor already quit or is broken')
        f: 'Future[_R]' = Future()
        work_item = _WorkItem(f, fn, *args, **kwargs)
        self._work_queue.put(work_item)
        return f
    if not TYPE_CHECKING:

        def submit(self, fn, *args, **kwargs):
            return self._submit(fn, *args, **kwargs)