import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_priority_filter(self):
    gdb = Recognizer(db=[self.Ninvar, self.Nbound])
    n = gdb.filter(priority=50)
    self.assertEqual(len(n), 0)
    gdb.add_gesture('T', [TGesture], priority=51)
    n = gdb.filter(priority=50)
    self.assertEqual(len(n), 0)
    n = gdb.filter(priority=51)
    self.assertEqual(len(n), 1)
    gdb.add_gesture('T', [TGesture], priority=52)
    n = gdb.filter(priority=[0, 51])
    self.assertEqual(len(n), 1)
    n = gdb.filter(priority=[0, 52])
    self.assertEqual(len(n), 2)
    n = gdb.filter(priority=[51, 52])
    self.assertEqual(len(n), 2)
    n = gdb.filter(priority=[52, 53])
    self.assertEqual(len(n), 1)
    n = gdb.filter(priority=[53, 54])
    self.assertEqual(len(n), 0)