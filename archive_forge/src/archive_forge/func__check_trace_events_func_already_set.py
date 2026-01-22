from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def _check_trace_events_func_already_set(self, glet):
    actions = self._trace_switch(glet)
    self.assertEqual(actions, [('return', '__enter__'), ('c_call', '_trace_switch'), ('call', 'run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('return', 'run'), ('c_return', '_trace_switch'), ('call', '__exit__'), ('c_call', '__exit__')])