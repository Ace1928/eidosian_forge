from unittest import mock
from oslo_config import cfg
from oslo_context import context as oslo_context
from testtools import matchers
from neutron_lib import context
from neutron_lib.tests import _base
def test_to_policy_values(self):
    values = {'user_id': 'user_id', 'tenant_id': 'tenant_id', 'is_admin': 'is_admin', 'tenant_name': 'tenant_name', 'user_name': 'user_name', 'domain_id': 'domain', 'user_domain_id': 'user_domain', 'project_domain_id': 'project_domain'}
    additional_values = {'user': 'user_id', 'tenant': 'tenant_id', 'project_id': 'tenant_id', 'project_name': 'tenant_name'}
    ctx = context.Context(**values)
    policy_values = dict(ctx.to_policy_values())
    self.assertDictSupersetOf(values, policy_values)
    self.assertDictSupersetOf(additional_values, policy_values)