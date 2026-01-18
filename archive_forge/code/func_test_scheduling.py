import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_scheduling(self):
    global best_score
    from kivy.clock import Clock
    gdb = Recognizer(db=[self.Tinvar, self.Ninvar])
    r = gdb.recognize([Ncandidate], max_gpf=1)
    r.bind(on_complete=best_score_cb)
    Clock.tick()
    self.assertEqual(r.progress, 0.5)
    self.assertEqual(best_score, 0.0)
    Clock.tick()
    self.assertEqual(r.progress, 1)
    self.assertTrue(best_score > 0.94 and best_score < 0.95)