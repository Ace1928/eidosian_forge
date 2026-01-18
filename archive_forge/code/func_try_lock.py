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
def try_lock():
    try:
        my_lock = pl.InterProcessLock(lock_file)
        my_lock.lockfile = open(lock_file, 'w')
        my_lock.trylock()
        my_lock.unlock()
        os._exit(1)
    except IOError:
        os._exit(0)