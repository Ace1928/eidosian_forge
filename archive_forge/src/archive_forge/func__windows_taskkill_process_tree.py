import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _windows_taskkill_process_tree(pid):
    try:
        subprocess.check_output(['taskkill', '/F', '/T', '/PID', str(pid)], stderr=None)
    except subprocess.CalledProcessError as e:
        if e.returncode not in [128, 255]:
            raise