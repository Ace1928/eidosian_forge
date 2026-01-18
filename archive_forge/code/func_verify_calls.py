import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
def verify_calls(self, mock_step):
    actual_calls = mock_step.mock_calls
    CallList = type(actual_calls)
    idx = 0
    try:
        for group in self.expected_groups:
            group_len = len(group)
            group_actual = CallList(actual_calls[idx:idx + group_len])
            idx += group_len
            mock_step.mock_calls = group_actual
            mock_step.assert_has_calls([mock.call(s, t) for s, t in group], any_order=True)
    finally:
        mock_step.actual_calls = actual_calls
    if len(actual_calls) > idx:
        raise AssertionError('Unexpected calls: %s' % CallList(actual_calls[idx:]))