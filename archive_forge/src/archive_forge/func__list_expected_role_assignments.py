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
def _list_expected_role_assignments(self, **filters):
    """Given the filters, it returns expected direct role assignments.

        :param filters: filters that will be considered when listing role
                        assignments. Valid filters are: role_id, domain_id,
                        project_id, group_id, user_id and
                        inherited_to_projects.

        :returns: the list of the expected role assignments.

        """
    assignment_link = self.build_role_assignment_link(**filters)
    user_ids = [None]
    if filters.get('group_id'):
        user_ids = [user['id'] for user in PROVIDERS.identity_api.list_users_in_group(filters['group_id'])]
    else:
        user_ids = [self.default_user_id]
    project_ids = [None]
    if filters.get('domain_id'):
        project_ids = [project['id'] for project in PROVIDERS.resource_api.list_projects_in_domain(filters.pop('domain_id'))]
    else:
        project_ids = [project['id'] for project in PROVIDERS.resource_api.list_projects_in_subtree(self.project_id)]
    assignments = []
    for project_id in project_ids:
        filters['project_id'] = project_id
        for user_id in user_ids:
            filters['user_id'] = user_id
            assignments.append(self.build_role_assignment_entity(link=assignment_link, **filters))
    return assignments