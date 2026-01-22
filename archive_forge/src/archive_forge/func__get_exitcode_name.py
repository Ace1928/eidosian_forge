import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _get_exitcode_name(exitcode):
    if sys.platform == 'win32':
        return 'UNKNOWN'
    if exitcode < 0:
        try:
            import signal
            return signal.Signals(-exitcode).name
        except ValueError:
            return 'UNKNOWN'
    elif exitcode != 255:
        return 'EXIT'
    return 'UNKNOWN'