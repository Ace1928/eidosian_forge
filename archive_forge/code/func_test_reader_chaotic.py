import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def test_reader_chaotic(self):
    lock = fasteners.ReaderWriterLock()
    activated = collections.deque()

    def chaotic_reader(blow_up):
        with lock.read_lock():
            if blow_up:
                raise RuntimeError('Broken')
            else:
                activated.append(lock.owner)

    def happy_writer():
        with lock.write_lock():
            activated.append(lock.owner)
    with futures.ThreadPoolExecutor(max_workers=20) as e:
        for i in range(0, 20):
            if i % 2 == 0:
                e.submit(chaotic_reader, blow_up=bool(i % 4 == 0))
            else:
                e.submit(happy_writer)
    writers = [a for a in activated if a == 'w']
    readers = [a for a in activated if a == 'r']
    self.assertEqual(10, len(writers))
    self.assertEqual(5, len(readers))