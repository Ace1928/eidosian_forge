import os
import subprocess
from unittest import mock
import uuid
from oslo_policy import policy as common_policy
from keystone.common import policies
from keystone.common.rbac_enforcer import policy
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class PolicyScopeTypesEnforcementTestCase(unit.TestCase):

    def setUp(self):
        super(PolicyScopeTypesEnforcementTestCase, self).setUp()
        rule = common_policy.RuleDefault(name='foo', check_str='', scope_types=['system'])
        policy._ENFORCER._enforcer.register_default(rule)
        self.credentials = {}
        self.action = 'foo'
        self.target = {}

    def test_forbidden_is_raised_if_enforce_scope_is_true(self):
        self.config_fixture.config(group='oslo_policy', enforce_scope=True)
        self.assertRaises(exception.ForbiddenAction, policy.enforce, self.credentials, self.action, self.target)

    def test_warning_message_is_logged_if_enforce_scope_is_false(self):
        self.config_fixture.config(group='oslo_policy', enforce_scope=False)
        expected_msg = 'Policy "foo": "" failed scope check. The token used to make the request was project scoped but the policy requires [\'system\'] scope. This behavior may change in the future where using the intended scope is required'
        with mock.patch('warnings.warn') as mock_warn:
            policy.enforce(self.credentials, self.action, self.target)
            mock_warn.assert_called_with(expected_msg)