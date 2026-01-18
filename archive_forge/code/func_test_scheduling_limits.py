import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_scheduling_limits(self):
    global best_score
    from kivy.clock import Clock
    gdb = Recognizer(db=[self.Ninvar])
    tpls = len(self.Ninvar.templates)
    best_score = 0
    gdb.db.append(self.Ninvar)
    r = gdb.recognize([Ncandidate], max_gpf=1)
    r.bind(on_complete=best_score_cb)
    self.assertEqual(r.progress, 0)
    Clock.tick()
    self.assertEqual(r.progress, 0.5)
    self.assertEqual(best_score, 0)
    Clock.tick()
    self.assertEqual(r.progress, 1)
    self.assertTrue(best_score > 0.94 and best_score < 0.95)
    best_score = 0
    gdb.db.append(self.Ninvar)
    r = gdb.recognize([Ncandidate], max_gpf=1)
    r.bind(on_complete=best_score_cb)
    self.assertEqual(r.progress, 0)
    Clock.tick()
    self.assertEqual(r.progress, 1 / 3.0)
    Clock.tick()
    self.assertEqual(r.progress, 2 / 3.0)
    self.assertEqual(best_score, 0)
    Clock.tick()
    self.assertEqual(r.progress, 1)
    self.assertTrue(best_score > 0.94 and best_score < 0.95)