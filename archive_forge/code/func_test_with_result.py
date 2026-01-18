import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_with_result(self, mock_start, mock_stop):
    self.assertEqual((1, 2), trace_with_result_func(1, i=2))
    start_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.trace_with_result_func', 'args': str((1,)), 'kwargs': str({'i': 2})}}
    stop_info = {'function': {'result': str((1, 2))}}
    mock_start.assert_called_once_with('hide_result', info=start_info)
    mock_stop.assert_called_once_with(info=stop_info)