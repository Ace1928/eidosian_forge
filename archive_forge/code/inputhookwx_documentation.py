import sys
import signal
from _pydev_bundle._pydev_saved_modules import time
from timeit import default_timer as clock
import wx
from pydev_ipython.inputhook import stdin_ready
Run the wx event loop by processing pending events only.

    This is like inputhook_wx1, but it keeps processing pending events
    until stdin is ready.  After processing all pending events, a call to
    time.sleep is inserted.  This is needed, otherwise, CPU usage is at 100%.
    This sleep time should be tuned though for best performance.
    