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
def _do_test_lock_externally(self, lock_dir):
    lock_path = os.path.join(lock_dir, 'lock')

    def lock_files(handles_dir):
        with pl.InterProcessLock(lock_path):
            handles = []
            for n in range(50):
                path = os.path.join(handles_dir, 'file-%s' % n)
                handles.append(open(path, 'w'))
            count = 0
            for handle in handles:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    count += 1
                    fcntl.flock(handle, fcntl.LOCK_UN)
                except IOError:
                    os._exit(2)
                finally:
                    handle.close()
            self.assertEqual(50, count)
    handles_dir = tempfile.mkdtemp()
    self.tmp_dirs.append(handles_dir)
    children = []
    for n in range(50):
        pid = os.fork()
        if pid:
            children.append(pid)
        else:
            try:
                lock_files(handles_dir)
            finally:
                os._exit(0)
    for child in children:
        pid, status = os.waitpid(child, 0)
        if pid:
            self.assertEqual(0, status)