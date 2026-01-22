import threading
from queue import Queue
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
class TestSelectDevice(ContextResettingTestCase):

    def test_select_device(self):
        exception_queue = Queue()
        for i in range(10):
            t = threading.Thread(target=newthread, args=(exception_queue,))
            t.start()
            t.join()
        exceptions = []
        while not exception_queue.empty():
            exceptions.append(exception_queue.get())
        self.assertEqual(exceptions, [])