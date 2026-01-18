from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy.tests import base
def test_wrap_check(self):
    state = _parser.ParseState()
    result = state._wrap_check('(', 'the_check', ')')
    self.assertEqual([('check', 'the_check')], result)