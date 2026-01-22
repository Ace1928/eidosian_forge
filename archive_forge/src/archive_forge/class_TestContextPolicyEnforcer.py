from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
class TestContextPolicyEnforcer(base.IsolatedUnitTest):

    def _do_test_policy_influence_context_admin(self, policy_admin_role, context_role, context_is_admin, admin_expected):
        self.config(policy_file=os.path.join(self.test_dir, 'gobble.gobble'), group='oslo_policy')
        rules = {'context_is_admin': 'role:%s' % policy_admin_role}
        self.set_policy_rules(rules)
        enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
        context = glance.context.RequestContext(roles=[context_role], is_admin=context_is_admin, policy_enforcer=enforcer)
        self.assertEqual(admin_expected, context.is_admin)

    def test_context_admin_policy_admin(self):
        self._do_test_policy_influence_context_admin('test_admin', 'test_admin', True, True)

    def test_context_nonadmin_policy_admin(self):
        self._do_test_policy_influence_context_admin('test_admin', 'test_admin', False, True)

    def test_context_admin_policy_nonadmin(self):
        self._do_test_policy_influence_context_admin('test_admin', 'demo', True, True)

    def test_context_nonadmin_policy_nonadmin(self):
        self._do_test_policy_influence_context_admin('test_admin', 'demo', False, False)