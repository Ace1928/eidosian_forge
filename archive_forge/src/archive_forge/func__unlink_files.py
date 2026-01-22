import atexit
import os
import signal
import sys
import ovs.vlog
def _unlink_files():
    for file_ in _files:
        if sys.platform == 'win32' and _files[file_]:
            _files[file_].close()
        _unlink(file_)