import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_writer_abort(self):
    lock = fasteners.ReaderWriterLock()
    self.assertFalse(lock.owner)

    def blow_up():
        with lock.write_lock():
            self.assertEqual(lock.WRITER, lock.owner)
            raise RuntimeError('Broken')
    self.assertRaises(RuntimeError, blow_up)
    self.assertFalse(lock.owner)