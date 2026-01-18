from __future__ import print_function
import sys
import greenlet
import unittest
from . import TestCase
from . import PY312
def test_trace_events_into_greenlet_func_already_set(self):

    def run():
        return tpt_callback()
    self._check_trace_events_func_already_set(greenlet.greenlet(run))