import os
import platform
import signal
import time
import warnings
from _thread import interrupt_main  # Py 3
from threading import Thread
from traitlets.log import get_logger
class ParentPollerUnix(Thread):
    """A Unix-specific daemon thread that terminates the program immediately
    when the parent process no longer exists.
    """

    def __init__(self):
        """Initialize the poller."""
        super().__init__()
        self.daemon = True

    def run(self):
        """Run the poller."""
        from errno import EINTR
        while True:
            try:
                if os.getppid() == 1:
                    get_logger().warning('Parent appears to have exited, shutting down.')
                    os._exit(1)
                time.sleep(1.0)
            except OSError as e:
                if e.errno == EINTR:
                    continue
                raise