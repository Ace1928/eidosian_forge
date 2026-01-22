import errno
import threading
from time import sleep
import weakref
def _acquire_non_blocking(acquire, timeout, retry_period, path):
    if retry_period is None:
        retry_period = 0.05
    start_time = get_time()
    while True:
        success = acquire()
        if success:
            return
        elif timeout is not None and get_time() - start_time > timeout:
            raise LockError("Couldn't lock {0}".format(path))
        else:
            sleep(retry_period)