import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def test_tmap__wait(self):
    r = range(1000)
    wq, results = tmap(lambda x: x, r, num_workers=5, wait=False)
    wq.wait()
    r2 = map(lambda x: x.result, results)
    self.assertEqual(list(r), list(r2))