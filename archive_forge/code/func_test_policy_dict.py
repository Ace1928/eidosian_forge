import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_policy_dict(self):
    user = uuid.uuid4().hex
    user_domain = uuid.uuid4().hex
    project_id = uuid.uuid4().hex
    project_domain = uuid.uuid4().hex
    roles = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex]
    service_user_id = uuid.uuid4().hex
    service_project_id = uuid.uuid4().hex
    service_roles = [uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex]
    ctx = context.RequestContext(user=user, user_domain=user_domain, project_id=project_id, project_domain=project_domain, roles=roles, service_user_id=service_user_id, service_project_id=service_project_id, service_roles=service_roles)
    self.assertEqual({'user_id': user, 'user_domain_id': user_domain, 'system_scope': None, 'domain_id': None, 'project_id': project_id, 'project_domain_id': project_domain, 'roles': roles, 'is_admin_project': True, 'service_user_id': service_user_id, 'service_user_domain_id': None, 'service_project_id': service_project_id, 'service_project_domain_id': None, 'service_roles': service_roles}, ctx.to_policy_values())
    system_all = 'all'
    ctx = context.RequestContext(user=user, user_domain=user_domain, system_scope=system_all, roles=roles, service_user_id=service_user_id, service_project_id=service_project_id, service_roles=service_roles)
    self.assertEqual({'user_id': user, 'user_domain_id': user_domain, 'system_scope': system_all, 'domain_id': None, 'project_id': None, 'project_domain_id': None, 'roles': roles, 'is_admin_project': True, 'service_user_id': service_user_id, 'service_user_domain_id': None, 'service_project_id': service_project_id, 'service_project_domain_id': None, 'service_roles': service_roles}, ctx.to_policy_values())
    domain_id = uuid.uuid4().hex
    ctx = context.RequestContext(user=user, user_domain=user_domain, domain_id=domain_id, roles=roles, service_user_id=service_user_id, service_project_id=service_project_id, service_roles=service_roles)
    self.assertEqual({'user_id': user, 'user_domain_id': user_domain, 'system_scope': None, 'domain_id': domain_id, 'project_id': None, 'project_domain_id': None, 'roles': roles, 'is_admin_project': True, 'service_user_id': service_user_id, 'service_user_domain_id': None, 'service_project_id': service_project_id, 'service_project_domain_id': None, 'service_roles': service_roles}, ctx.to_policy_values())
    ctx = context.RequestContext(user=user, user_domain=user_domain, project_id=project_id, project_domain=project_domain, roles=roles, is_admin_project=False, service_user_id=service_user_id, service_project_id=service_project_id, service_roles=service_roles)
    self.assertEqual({'user_id': user, 'user_domain_id': user_domain, 'system_scope': None, 'domain_id': None, 'project_id': project_id, 'project_domain_id': project_domain, 'roles': roles, 'is_admin_project': False, 'service_user_id': service_user_id, 'service_user_domain_id': None, 'service_project_id': service_project_id, 'service_project_domain_id': None, 'service_roles': service_roles}, ctx.to_policy_values())