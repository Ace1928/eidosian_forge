from __future__ import annotations
import atexit
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import weakref
from logging import Logger
from shutil import which as _which
from typing import Any
from tornado import gen
def _create_process(self, **kwargs: Any) -> subprocess.Popen[str]:
    """Create the watcher helper process."""
    kwargs['bufsize'] = 0
    if pty is not None:
        master, slave = pty.openpty()
        kwargs['stderr'] = kwargs['stdout'] = slave
        kwargs['start_new_session'] = True
        self._stdout = os.fdopen(master, 'rb')
    else:
        kwargs['stdout'] = subprocess.PIPE
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            kwargs['startupinfo'] = startupinfo
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
            kwargs['shell'] = True
    return super()._create_process(**kwargs)