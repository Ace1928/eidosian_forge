import os
import platform
import signal
import time
import warnings
from _thread import interrupt_main  # Py 3
from threading import Thread
from traitlets.log import get_logger
class ParentPollerWindows(Thread):
    """A Windows-specific daemon thread that listens for a special event that
    signals an interrupt and, optionally, terminates the program immediately
    when the parent process no longer exists.
    """

    def __init__(self, interrupt_handle=None, parent_handle=None):
        """Create the poller. At least one of the optional parameters must be
        provided.

        Parameters
        ----------
        interrupt_handle : HANDLE (int), optional
            If provided, the program will generate a Ctrl+C event when this
            handle is signaled.
        parent_handle : HANDLE (int), optional
            If provided, the program will terminate immediately when this
            handle is signaled.
        """
        assert interrupt_handle or parent_handle
        super().__init__()
        if ctypes is None:
            msg = 'ParentPollerWindows requires ctypes'
            raise ImportError(msg)
        self.daemon = True
        self.interrupt_handle = interrupt_handle
        self.parent_handle = parent_handle

    def run(self):
        """Run the poll loop. This method never returns."""
        try:
            from _winapi import INFINITE, WAIT_OBJECT_0
        except ImportError:
            from _subprocess import INFINITE, WAIT_OBJECT_0
        handles = []
        if self.interrupt_handle:
            handles.append(self.interrupt_handle)
        if self.parent_handle:
            handles.append(self.parent_handle)
        arch = platform.architecture()[0]
        c_int = ctypes.c_int64 if arch.startswith('64') else ctypes.c_int
        while True:
            result = ctypes.windll.kernel32.WaitForMultipleObjects(len(handles), (c_int * len(handles))(*handles), False, INFINITE)
            if WAIT_OBJECT_0 <= result < len(handles):
                handle = handles[result - WAIT_OBJECT_0]
                if handle == self.interrupt_handle:
                    if callable(signal.getsignal(signal.SIGINT)):
                        interrupt_main()
                elif handle == self.parent_handle:
                    get_logger().warning('Parent appears to have exited, shutting down.')
                    os._exit(1)
            elif result < 0:
                warnings.warn('Parent poll failed.  If the frontend dies,\n                the kernel may be left running.  Please let us know\n                about your system (bitness, Python, etc.) at\n                ipython-dev@scipy.org', stacklevel=2)
                return