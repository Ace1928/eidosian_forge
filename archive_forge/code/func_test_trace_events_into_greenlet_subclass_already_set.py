from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_trace_events_into_greenlet_subclass_already_set(self):

    class X(greenlet.greenlet):

        def run(self):
            return tpt_callback()
    self._check_trace_events_func_already_set(X())