from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_result_empty(self):
    state = _parser.ParseState()
    self.assertRaises(ValueError, lambda: state.result)