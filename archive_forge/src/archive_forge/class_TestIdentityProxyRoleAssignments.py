import uuid
from openstack.identity.v3 import _proxy
from openstack.identity.v3 import access_rule
from openstack.identity.v3 import credential
from openstack.identity.v3 import domain
from openstack.identity.v3 import domain_config
from openstack.identity.v3 import endpoint
from openstack.identity.v3 import group
from openstack.identity.v3 import policy
from openstack.identity.v3 import project
from openstack.identity.v3 import region
from openstack.identity.v3 import role
from openstack.identity.v3 import role_domain_group_assignment
from openstack.identity.v3 import role_domain_user_assignment
from openstack.identity.v3 import role_project_group_assignment
from openstack.identity.v3 import role_project_user_assignment
from openstack.identity.v3 import role_system_group_assignment
from openstack.identity.v3 import role_system_user_assignment
from openstack.identity.v3 import service
from openstack.identity.v3 import trust
from openstack.identity.v3 import user
from openstack.tests.unit import test_proxy_base
class TestIdentityProxyRoleAssignments(TestIdentityProxyBase):

    def test_role_assignments_filter__domain_user(self):
        self.verify_list(self.proxy.role_assignments_filter, role_domain_user_assignment.RoleDomainUserAssignment, method_kwargs={'domain': 'domain', 'user': 'user'}, expected_kwargs={'domain_id': 'domain', 'user_id': 'user'})

    def test_role_assignments_filter__domain_group(self):
        self.verify_list(self.proxy.role_assignments_filter, role_domain_group_assignment.RoleDomainGroupAssignment, method_kwargs={'domain': 'domain', 'group': 'group'}, expected_kwargs={'domain_id': 'domain', 'group_id': 'group'})

    def test_role_assignments_filter__project_user(self):
        self.verify_list(self.proxy.role_assignments_filter, role_project_user_assignment.RoleProjectUserAssignment, method_kwargs={'project': 'project', 'user': 'user'}, expected_kwargs={'project_id': 'project', 'user_id': 'user'})

    def test_role_assignments_filter__project_group(self):
        self.verify_list(self.proxy.role_assignments_filter, role_project_group_assignment.RoleProjectGroupAssignment, method_kwargs={'project': 'project', 'group': 'group'}, expected_kwargs={'project_id': 'project', 'group_id': 'group'})

    def test_role_assignments_filter__system_user(self):
        self.verify_list(self.proxy.role_assignments_filter, role_system_user_assignment.RoleSystemUserAssignment, method_kwargs={'system': 'system', 'user': 'user'}, expected_kwargs={'system_id': 'system', 'user_id': 'user'})

    def test_role_assignments_filter__system_group(self):
        self.verify_list(self.proxy.role_assignments_filter, role_system_group_assignment.RoleSystemGroupAssignment, method_kwargs={'system': 'system', 'group': 'group'}, expected_kwargs={'system_id': 'system', 'group_id': 'group'})

    def test_assign_domain_role_to_user(self):
        self._verify('openstack.identity.v3.domain.Domain.assign_role_to_user', self.proxy.assign_domain_role_to_user, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_domain_role_from_user(self):
        self._verify('openstack.identity.v3.domain.Domain.unassign_role_from_user', self.proxy.unassign_domain_role_from_user, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_user_has_domain_role(self):
        self._verify('openstack.identity.v3.domain.Domain.validate_user_has_role', self.proxy.validate_user_has_domain_role, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_assign_domain_role_to_group(self):
        self._verify('openstack.identity.v3.domain.Domain.assign_role_to_group', self.proxy.assign_domain_role_to_group, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_domain_role_from_group(self):
        self._verify('openstack.identity.v3.domain.Domain.unassign_role_from_group', self.proxy.unassign_domain_role_from_group, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_group_has_domain_role(self):
        self._verify('openstack.identity.v3.domain.Domain.validate_group_has_role', self.proxy.validate_group_has_domain_role, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_assign_project_role_to_user(self):
        self._verify('openstack.identity.v3.project.Project.assign_role_to_user', self.proxy.assign_project_role_to_user, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_project_role_from_user(self):
        self._verify('openstack.identity.v3.project.Project.unassign_role_from_user', self.proxy.unassign_project_role_from_user, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_user_has_project_role(self):
        self._verify('openstack.identity.v3.project.Project.validate_user_has_role', self.proxy.validate_user_has_project_role, method_args=['dom_id'], method_kwargs={'user': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_assign_project_role_to_group(self):
        self._verify('openstack.identity.v3.project.Project.assign_role_to_group', self.proxy.assign_project_role_to_group, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_project_role_from_group(self):
        self._verify('openstack.identity.v3.project.Project.unassign_role_from_group', self.proxy.unassign_project_role_from_group, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_group_has_project_role(self):
        self._verify('openstack.identity.v3.project.Project.validate_group_has_role', self.proxy.validate_group_has_project_role, method_args=['dom_id'], method_kwargs={'group': 'uid', 'role': 'rid'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_assign_system_role_to_user(self):
        self._verify('openstack.identity.v3.system.System.assign_role_to_user', self.proxy.assign_system_role_to_user, method_kwargs={'user': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_system_role_from_user(self):
        self._verify('openstack.identity.v3.system.System.unassign_role_from_user', self.proxy.unassign_system_role_from_user, method_kwargs={'user': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_user_has_system_role(self):
        self._verify('openstack.identity.v3.system.System.validate_user_has_role', self.proxy.validate_user_has_system_role, method_kwargs={'user': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(user.User, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_assign_system_role_to_group(self):
        self._verify('openstack.identity.v3.system.System.assign_role_to_group', self.proxy.assign_system_role_to_group, method_kwargs={'group': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_unassign_system_role_from_group(self):
        self._verify('openstack.identity.v3.system.System.unassign_role_from_group', self.proxy.unassign_system_role_from_group, method_kwargs={'group': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])

    def test_validate_group_has_system_role(self):
        self._verify('openstack.identity.v3.system.System.validate_group_has_role', self.proxy.validate_group_has_system_role, method_kwargs={'group': 'uid', 'role': 'rid', 'system': 'all'}, expected_args=[self.proxy, self.proxy._get_resource(group.Group, 'uid'), self.proxy._get_resource(role.Role, 'rid')])