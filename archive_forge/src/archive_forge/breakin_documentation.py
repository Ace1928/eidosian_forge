import os
import signal
from typing import Optional
Add a signal handler so we drop into the debugger.

    On Unix, this is hooked into SIGQUIT (C-\), and on Windows, this is
    hooked into SIGBREAK (C-Pause).
    