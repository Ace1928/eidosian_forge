import pytest
import unittest
import kivy.multistroke
from kivy.multistroke import Recognizer, MultistrokeGesture
from kivy.vector import Vector
def test_protractor_bound(self):
    gdb = Recognizer(db=[self.Tbound, self.Nbound])
    r = gdb.recognize([NGesture], orientation_sensitive=True, max_gpf=0)
    self.assertEqual(r.best['name'], 'N')
    self.assertTrue(r.best['score'] >= 0.99)
    r = gdb.recognize([NGesture], orientation_sensitive=False, max_gpf=0)
    self.assertEqual(r.best['name'], None)
    self.assertEqual(r.best['score'], 0)
    r = gdb.recognize([Ncandidate], orientation_sensitive=True, max_gpf=0)
    self.assertEqual(r.best['name'], 'N')
    self.assertTrue(r.best['score'] > 0.94 and r.best['score'] < 0.95)