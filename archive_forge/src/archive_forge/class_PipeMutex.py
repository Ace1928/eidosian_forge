import errno
import fcntl
import os
import eventlet
import eventlet.debug
import eventlet.greenthread
import eventlet.hubs
class PipeMutex(object):
    """Mutex using a pipe.

    Works across both greenlets and real threads, even at the same time.

    Class code copied from Swift's swift/common/utils.py
    Related eventlet bug: https://github.com/eventlet/eventlet/issues/432
    """

    def __init__(self):
        self.rfd, self.wfd = os.pipe()
        rflags = fcntl.fcntl(self.rfd, fcntl.F_GETFL)
        fcntl.fcntl(self.rfd, fcntl.F_SETFL, rflags | os.O_NONBLOCK)
        os.write(self.wfd, b'-')
        self.owner = None
        self.recursion_depth = 0
        eventlet.debug.hub_prevent_multiple_readers(False)

    def acquire(self, blocking=True):
        """Acquire the mutex.

        If called with blocking=False, returns True if the mutex was
        acquired and False if it wasn't. Otherwise, blocks until the mutex
        is acquired and returns True.
        This lock is recursive; the same greenthread may acquire it as many
        times as it wants to, though it must then release it that many times
        too.
        """
        current_greenthread_id = id(eventlet.greenthread.getcurrent())
        if self.owner == current_greenthread_id:
            self.recursion_depth += 1
            return True
        while True:
            try:
                os.read(self.rfd, 1)
                self.owner = current_greenthread_id
                return True
            except OSError as err:
                if err.errno != errno.EAGAIN:
                    raise
                if not blocking:
                    return False
                eventlet.hubs.trampoline(self.rfd, read=True)

    def release(self):
        """Release the mutex."""
        current_greenthread_id = id(eventlet.greenthread.getcurrent())
        if self.owner != current_greenthread_id:
            raise RuntimeError('cannot release un-acquired lock')
        if self.recursion_depth > 0:
            self.recursion_depth -= 1
            return
        self.owner = None
        os.write(self.wfd, b'X')

    def close(self):
        """Close the mutex.

        This releases its file descriptors.
        You can't use a mutex after it's been closed.
        """
        if self.wfd is not None:
            os.close(self.wfd)
            self.wfd = None
        if self.rfd is not None:
            os.close(self.rfd)
            self.rfd = None
        self.owner = None
        self.recursion_depth = 0

    def __del__(self):
        self.close()