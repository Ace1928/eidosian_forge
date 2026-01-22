import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
class RoleAssignmentDirectTestCase(RoleAssignmentBaseTestCase):
    """Class for testing direct assignments on /v3/role_assignments API.

    Direct assignments on a domain or project have effect on them directly,
    instead of on their project hierarchy, i.e they are non-inherited. In
    addition, group direct assignments are not expanded to group's users.

    Tests on this class make assertions on the representation and API filtering
    of direct assignments.

    """

    def _test_get_role_assignments(self, **filters):
        """Generic filtering test method.

        According to the provided filters, this method:
        - creates a new role assignment;
        - asserts that list role assignments API reponds correctly;
        - deletes the created role assignment.

        :param filters: filters to be considered when listing role assignments.
                        Valid filters are: role_id, domain_id, project_id,
                        group_id, user_id and inherited_to_projects.

        """
        test_assignment = self._set_default_assignment_attributes(**filters)
        PROVIDERS.assignment_api.create_grant(**test_assignment)
        expected_assignments = self._list_expected_role_assignments(**test_assignment)
        response, query_url = self.get_role_assignments(**test_assignment)
        self.assertValidRoleAssignmentListResponse(response, resource_url=query_url)
        self.assertEqual(len(expected_assignments), len(response.result.get('role_assignments')))
        for assignment in expected_assignments:
            self.assertRoleAssignmentInListResponse(response, assignment)
        PROVIDERS.assignment_api.delete_grant(**test_assignment)

    def _set_default_assignment_attributes(self, **attribs):
        """Insert default values for missing attributes of role assignment.

        If no actor, target or role are provided, they will default to values
        from sample data.

        :param attribs: info from a role assignment entity. Valid attributes
                        are: role_id, domain_id, project_id, group_id, user_id
                        and inherited_to_projects.

        """
        if not any((target in attribs for target in ('domain_id', 'projects_id'))):
            attribs['project_id'] = self.project_id
        if not any((actor in attribs for actor in ('user_id', 'group_id'))):
            attribs['user_id'] = self.default_user_id
        if 'role_id' not in attribs:
            attribs['role_id'] = self.role_id
        return attribs

    def _list_expected_role_assignments(self, **filters):
        """Given the filters, it returns expected direct role assignments.

        :param filters: filters that will be considered when listing role
                        assignments. Valid filters are: role_id, domain_id,
                        project_id, group_id, user_id and
                        inherited_to_projects.

        :returns: the list of the expected role assignments.

        """
        return [self.build_role_assignment_entity(**filters)]

    def test_get_role_assignments_by_domain(self, **filters):
        self._test_get_role_assignments(domain_id=self.domain_id, **filters)

    def test_get_role_assignments_by_project(self, **filters):
        self._test_get_role_assignments(project_id=self.project_id, **filters)

    def test_get_role_assignments_by_user(self, **filters):
        self._test_get_role_assignments(user_id=self.default_user_id, **filters)

    def test_get_role_assignments_by_group(self, **filters):
        self._test_get_role_assignments(group_id=self.default_group_id, **filters)

    def test_get_role_assignments_by_role(self, **filters):
        self._test_get_role_assignments(role_id=self.role_id, **filters)

    def test_get_role_assignments_by_domain_and_user(self, **filters):
        self.test_get_role_assignments_by_domain(user_id=self.default_user_id, **filters)

    def test_get_role_assignments_by_domain_and_group(self, **filters):
        self.test_get_role_assignments_by_domain(group_id=self.default_group_id, **filters)

    def test_get_role_assignments_by_project_and_user(self, **filters):
        self.test_get_role_assignments_by_project(user_id=self.default_user_id, **filters)

    def test_get_role_assignments_by_project_and_group(self, **filters):
        self.test_get_role_assignments_by_project(group_id=self.default_group_id, **filters)

    def test_get_role_assignments_by_domain_user_and_role(self, **filters):
        self.test_get_role_assignments_by_domain_and_user(role_id=self.role_id, **filters)

    def test_get_role_assignments_by_domain_group_and_role(self, **filters):
        self.test_get_role_assignments_by_domain_and_group(role_id=self.role_id, **filters)

    def test_get_role_assignments_by_project_user_and_role(self, **filters):
        self.test_get_role_assignments_by_project_and_user(role_id=self.role_id, **filters)

    def test_get_role_assignments_by_project_group_and_role(self, **filters):
        self.test_get_role_assignments_by_project_and_group(role_id=self.role_id, **filters)