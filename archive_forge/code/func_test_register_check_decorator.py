from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
@mock.patch.object(_checks, 'registered_checks', {})
def test_register_check_decorator(self):

    @_checks.register('spam')
    class TestCheck(_checks.Check):
        pass
    self.assertEqual(dict(spam=TestCheck), _checks.registered_checks)