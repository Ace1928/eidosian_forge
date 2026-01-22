import code
import greenlet
import logging
import signal
from curtsies.input import is_main_thread
Fakes sys.stdout or sys.stderr

        on_write should always take unicode

        fileno should be the fileno that on_write will
                output to (e.g. 1 for standard output).
        