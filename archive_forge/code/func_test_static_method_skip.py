import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_static_method_skip(self, mock_start, mock_stop):
    self.assertEqual(25, FakeTraceStaticMethodSkip.static_method(25))
    self.assertFalse(mock_start.called)
    self.assertFalse(mock_stop.called)