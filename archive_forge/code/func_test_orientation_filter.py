import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_orientation_filter(self):
    gdb = Recognizer(db=[self.Ninvar, self.Nbound])
    n = gdb.filter(orientation_sensitive=True)
    self.assertEqual(len(n), 1)
    n = gdb.filter(orientation_sensitive=False)
    self.assertEqual(len(n), 1)
    n = gdb.filter(orientation_sensitive=None)
    self.assertEqual(len(n), 2)
    gdb.db.append(self.Tinvar)
    n = gdb.filter(orientation_sensitive=True)
    self.assertEqual(len(n), 1)
    n = gdb.filter(orientation_sensitive=False)
    self.assertEqual(len(n), 2)
    n = gdb.filter(orientation_sensitive=None)
    self.assertEqual(len(n), 3)