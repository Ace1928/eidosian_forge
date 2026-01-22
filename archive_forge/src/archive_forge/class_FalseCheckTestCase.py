from unittest import mock
from oslotest import base as test_base
from oslo_policy import _checks
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
class FalseCheckTestCase(test_base.BaseTestCase):

    def test_str(self):
        check = _checks.FalseCheck()
        self.assertEqual('!', str(check))

    def test_call(self):
        check = _checks.FalseCheck()
        self.assertFalse(check('target', 'creds', None))