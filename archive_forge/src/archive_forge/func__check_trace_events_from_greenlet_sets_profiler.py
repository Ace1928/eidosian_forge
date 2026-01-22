from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def _check_trace_events_from_greenlet_sets_profiler(self, g, tracer):
    g.switch()
    tpt_callback()
    tracer.__exit__()
    self.assertEqual(tracer.actions, [('return', '__enter__'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('return', 'run'), ('call', 'tpt_callback'), ('return', 'tpt_callback'), ('call', '__exit__'), ('c_call', '__exit__')])