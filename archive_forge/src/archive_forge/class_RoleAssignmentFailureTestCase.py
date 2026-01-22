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
class RoleAssignmentFailureTestCase(RoleAssignmentBaseTestCase):
    """Class for testing invalid query params on /v3/role_assignments API.

    Querying domain and project, or user and group results in a HTTP 400 Bad
    Request, since a role assignment must contain only a single pair of (actor,
    target). In addition, since filtering on role assignments applies only to
    the final result, effective mode cannot be combined with i) group or ii)
    domain and inherited, because it would always result in an empty list.

    """

    def test_get_role_assignments_by_domain_and_project(self):
        self.get_role_assignments(domain_id=self.domain_id, project_id=self.project_id, expected_status=http.client.BAD_REQUEST)

    def test_get_role_assignments_by_user_and_group(self):
        self.get_role_assignments(user_id=self.default_user_id, group_id=self.default_group_id, expected_status=http.client.BAD_REQUEST)

    def test_get_role_assignments_by_effective_and_inherited(self):
        self.get_role_assignments(domain_id=self.domain_id, effective=True, inherited_to_projects=True, expected_status=http.client.BAD_REQUEST)

    def test_get_role_assignments_by_effective_and_group(self):
        self.get_role_assignments(effective=True, group_id=self.default_group_id, expected_status=http.client.BAD_REQUEST)