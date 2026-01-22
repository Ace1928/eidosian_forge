import queue
import threading
import time
class FairSemaphore(object):
    """Semaphore class that notifies in order of request.

    We cannot use a normal Semaphore because it doesn't give any ordering,
    which could lead to a request starving. Instead, handle them in the
    order we receive them.

    :param int concurrency:
        How many concurrent threads can have the semaphore at once.
    :param float rate_delay:
        How long to wait between the start of each thread receiving the
        semaphore.
    """

    def __init__(self, concurrency, rate_delay):
        self._lock = threading.Lock()
        self._concurrency = concurrency
        if concurrency:
            self._count = 0
            self._queue = queue.Queue()
        self._rate_delay = rate_delay
        self._rate_last_ts = time.time()

    def __enter__(self):
        """Aquire a semaphore."""
        if not self._concurrency:
            with self._lock:
                execution_time = self._advance_timer()
        else:
            execution_time = self._get_ticket()
        return self._wait_for_execution(execution_time)

    def _wait_for_execution(self, execution_time):
        """Wait until the pre-calculated time to run."""
        wait_time = execution_time - time.time()
        if wait_time > 0:
            time.sleep(wait_time)

    def _get_ticket(self):
        ticket = threading.Event()
        with self._lock:
            if self._count <= self._concurrency:
                self._count += 1
                return self._advance_timer()
            else:
                self._queue.put(ticket)
        ticket.wait()
        with self._lock:
            return self._advance_timer()

    def _advance_timer(self):
        """Calculate the time when it's ok to run a command again.

        This runs inside of the mutex, serializing the calculation
        of when it's ok to run again and setting _rate_last_ts to that
        new time so that the next thread to calculate when it's safe to
        run starts from the time that the current thread calculated.
        """
        self._rate_last_ts = self._rate_last_ts + self._rate_delay
        return self._rate_last_ts

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the semaphore."""
        if not self._concurrency:
            return
        with self._lock:
            if self._queue.qsize() > 0:
                ticket = self._queue.get()
                ticket.set()
            else:
                self._count -= 1