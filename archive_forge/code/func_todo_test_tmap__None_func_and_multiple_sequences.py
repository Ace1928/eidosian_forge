import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def todo_test_tmap__None_func_and_multiple_sequences(self):
    """Using a None as func and multiple sequences"""
    self.fail()
    res = tmap(None, [1, 2, 3, 4])
    res2 = tmap(None, [1, 2, 3, 4], [22, 33, 44, 55])
    res3 = tmap(None, [1, 2, 3, 4], [22, 33, 44, 55, 66])
    res4 = tmap(None, [1, 2, 3, 4, 5], [22, 33, 44, 55])
    self.assertEqual([1, 2, 3, 4], res)
    self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55)], res2)
    self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (None, 66)], res3)
    self.assertEqual([(1, 22), (2, 33), (3, 44), (4, 55), (5, None)], res4)