import collections
import copy
import datetime
import re
from unittest import mock
from osprofiler import profiler
from osprofiler.tests import test
@mock.patch('osprofiler.profiler.stop')
@mock.patch('osprofiler.profiler.start')
def test_private_methods(self, mock_start, mock_stop):
    fake_cls = FakeTraceWithMetaclassPrivate()
    self.assertEqual(10, fake_cls._new_private_method(5))
    expected_info = {'function': {'name': 'osprofiler.tests.unit.test_profiler.FakeTraceWithMetaclassPrivate._new_private_method', 'args': str((fake_cls, 5)), 'kwargs': str({})}}
    self.assertEqual(1, len(mock_start.call_args_list))
    self.assertIn(mock_start.call_args_list[0], possible_mock_calls('rpc', expected_info))
    mock_stop.assert_called_once_with()