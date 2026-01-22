import os
import shlex
import shutil
import sys
import subprocess
import threading
import warnings
def _find_grail_rc(self):
    import glob
    import pwd
    import socket
    import tempfile
    tempdir = os.path.join(tempfile.gettempdir(), '.grail-unix')
    user = pwd.getpwuid(os.getuid())[0]
    filename = os.path.join(glob.escape(tempdir), glob.escape(user) + '-*')
    maybes = glob.glob(filename)
    if not maybes:
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    for fn in maybes:
        try:
            s.connect(fn)
        except OSError:
            try:
                os.unlink(fn)
            except OSError:
                pass
        else:
            return s