import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_without_args(self, mock_start, mock_stop):
    fake_cls = FakeTraceWithMetaclassHideArgs()
    self.assertEqual(20, fake_cls.method5(5, 15))
    expected_info = {'b': 20, 'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassHideArgs.method5'}}
    self.assertEqual(1, len(mock_start.call_args_list))
    self.assertIn(mock_start.call_args_list[0], possible_mock_calls('a', expected_info))
    mock_stop.assert_called_once_with()