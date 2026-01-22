from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class CheckRegisterTestCase(test_base.BaseTestCase):

    @mock.patch.object(_checks, 'registered_checks', {})
    def test_register_check(self):

        class TestCheck(_checks.Check):
            pass
        _checks.register('spam', TestCheck)
        self.assertEqual(dict(spam=TestCheck), _checks.registered_checks)

    @mock.patch.object(_checks, 'registered_checks', {})
    def test_register_check_decorator(self):

        @_checks.register('spam')
        class TestCheck(_checks.Check):
            pass
        self.assertEqual(dict(spam=TestCheck), _checks.registered_checks)