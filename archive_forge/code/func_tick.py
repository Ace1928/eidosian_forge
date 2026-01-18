import unittest
from collections import namedtuple
from bpython.curtsies import combined_events
from bpython.test import FixLanguageTestCase as TestCase
import curtsies.events
def tick(self, dt=1):
    self._current_tick += dt
    return self._current_tick