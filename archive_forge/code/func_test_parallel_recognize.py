import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_parallel_recognize(self):
    global counter
    from kivy.clock import Clock
    counter = 0
    gdb = Recognizer()
    for i in range(9):
        gdb.add_gesture('T', [TGesture], priority=50)
    gdb.add_gesture('N', [NGesture])
    r1 = gdb.recognize([Ncandidate], max_gpf=1)
    r1.bind(on_complete=counter_cb)
    Clock.tick()
    r2 = gdb.recognize([Ncandidate], max_gpf=1)
    r2.bind(on_complete=counter_cb)
    Clock.tick()
    r3 = gdb.recognize([Ncandidate], max_gpf=1)
    r3.bind(on_complete=counter_cb)
    Clock.tick()
    for i in range(5):
        n = gdb.recognize([TGesture], max_gpf=0)
        self.assertEqual(n.best['name'], 'T')
        self.assertTrue(round(n.best['score'], 1) == 1.0)
    for i in range(6):
        Clock.tick()
    self.assertEqual(counter, 0)
    Clock.tick()
    self.assertEqual(counter, 1)
    Clock.tick()
    self.assertEqual(counter, 2)
    Clock.tick()
    self.assertEqual(counter, 3)