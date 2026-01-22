import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
def _posix_recursive_kill(pid):
    """Recursively kill the descendants of a process before killing it."""
    try:
        children_pids = subprocess.check_output(['pgrep', '-P', str(pid)], stderr=None, text=True)
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            children_pids = ''
        else:
            raise
    for cpid in children_pids.splitlines():
        cpid = int(cpid)
        _posix_recursive_kill(cpid)
    _kill(pid)