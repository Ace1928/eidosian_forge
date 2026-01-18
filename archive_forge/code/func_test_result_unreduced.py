from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_result_unreduced(self):
    state = _parser.ParseState()
    state.tokens = ['tok1', 'tok2']
    state.values = ['val1', 'val2']
    self.assertRaises(ValueError, lambda: state.result)