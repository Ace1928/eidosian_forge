import errno
import fcntl
import os
import subprocess
import time
from . import Connection, ConnectionException
def _restore_display(self):
    if self._old_display is None:
        del os.environ['DISPLAY']
    else:
        os.environ['DISPLAY'] = self._old_display