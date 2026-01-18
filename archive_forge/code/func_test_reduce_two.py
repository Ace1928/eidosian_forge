from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok4'], 'meth2'), (['tok2', 'tok3'], 'meth1')])
@mock.patch.object(_parser.ParseState, 'meth1', create=True, return_value=[('tok4', 'val4')])
@mock.patch.object(_parser.ParseState, 'meth2', create=True, return_value=[('tok5', 'val5')])
def test_reduce_two(self, mock_meth2, mock_meth1):
    state = _parser.ParseState()
    state.tokens = ['tok1', 'tok2', 'tok3']
    state.values = ['val1', 'val2', 'val3']
    state.reduce()
    self.assertEqual(['tok5'], state.tokens)
    self.assertEqual(['val5'], state.values)
    mock_meth1.assert_called_once_with('val2', 'val3')
    mock_meth2.assert_called_once_with('val1', 'val4')