import atexit
import queue
import threading
import weakref
def _wait_for_work(self):
    self.idle = True
    work = None
    while work is None:
        try:
            work = self.work_queue.get(True, self.MAX_IDLE_FOR)
        except queue.Empty:
            if self._is_dying():
                work = _TOMBSTONE
    self.idle = False
    return work