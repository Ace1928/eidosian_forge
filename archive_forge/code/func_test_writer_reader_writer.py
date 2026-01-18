import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_writer_reader_writer(self):
    lock = fasteners.ReaderWriterLock()
    with lock.write_lock():
        self.assertTrue(lock.is_writer())
        with lock.read_lock():
            self.assertTrue(lock.is_reader())
            with lock.write_lock():
                self.assertTrue(lock.is_writer())