from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
@mock.patch.object(_parser.ParseState, 'reducers', [(['tok1', 'tok2'], 'meth')])
@mock.patch.object(_parser.ParseState, 'meth', create=True)
def test_reduce_short(self, mock_meth):
    state = _parser.ParseState()
    state.tokens = ['tok1']
    state.values = ['val1']
    state.reduce()
    self.assertEqual(['tok1'], state.tokens)
    self.assertEqual(['val1'], state.values)
    self.assertFalse(mock_meth.called)