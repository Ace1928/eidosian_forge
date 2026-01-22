import errno
import threading
from time import sleep
import weakref
def _lock_file_blocking(file_):
    fcntl.flock(file_.fileno(), fcntl.LOCK_EX)