import queue
import threading
import time
def _wait_for_execution(self, execution_time):
    """Wait until the pre-calculated time to run."""
    wait_time = execution_time - time.time()
    if wait_time > 0:
        time.sleep(wait_time)