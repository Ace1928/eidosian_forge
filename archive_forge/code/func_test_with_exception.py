import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_with_exception(self, mock_start, mock_stop):
    self.assertRaises(ValueError, test_fn_exc)
    expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.test_fn_exc'}}
    expected_stop_info = {'etype': 'ValueError', 'message': ''}
    mock_start.assert_called_once_with('foo', info=expected_info)
    mock_stop.assert_called_once_with(info=expected_stop_info)