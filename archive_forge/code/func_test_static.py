import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
@test.testcase.skip('Static method tracing was disabled due the bug. This test should be skipped until we find the way to address it.')
def test_static(self, mock_start, mock_stop):
    fake_cls = FakeTraceStaticMethod()
    self.assertEqual(25, fake_cls.static_method(25))
    expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profilerosprofiler.tests.unit.test_profiler.FakeTraceStatic.method4', 'args': str((25,)), 'kwargs': str({})}}
    self.assertEqual(1, len(mock_start.call_args_list))
    self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
    mock_stop.assert_called_once_with()