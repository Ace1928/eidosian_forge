import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_timeout_case_2(self):
    global best_score
    from kivy.clock import Clock
    from time import sleep
    best_score = 0
    gdb = Recognizer(db=[self.Tbound, self.Ninvar, self.Tinvar])
    r = gdb.recognize([Ncandidate], max_gpf=1, timeout=0.8)
    Clock.tick()
    self.assertEqual(best_score, 0)
    sleep(0.4)
    Clock.tick()
    sleep(0.4)
    Clock.tick()
    self.assertEqual(r.status, 'timeout')
    self.assertEqual(r.progress, 2 / 3.0)
    self.assertTrue(r.best['score'] >= 0.94 and r.best['score'] <= 0.95)