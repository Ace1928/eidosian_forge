import errno
import fcntl
import multiprocessing
import os
import shutil
import signal
import tempfile
import threading
import time
from fasteners import process_lock as pl
from fasteners import test
def trylock(self):
    err = IOError()
    err.errno = self.errno_code
    raise err