from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _get_system_roles(self):
    roles = []
    groups = PROVIDERS.identity_api.list_groups_for_user(self.user_id)
    all_group_roles = []
    assignments = []
    for group in groups:
        group_roles = PROVIDERS.assignment_api.list_system_grants_for_group(group['id'])
        for role in group_roles:
            all_group_roles.append(role)
            assignment = {'group_id': group['id'], 'role_id': role['id']}
            assignments.append(assignment)
    user_roles = PROVIDERS.assignment_api.list_system_grants_for_user(self.user_id)
    for role in user_roles:
        assignment = {'user_id': self.user_id, 'role_id': role['id']}
        assignments.append(assignment)
    assignments = PROVIDERS.assignment_api.add_implied_roles(assignments)
    for assignment in assignments:
        role = PROVIDERS.role_api.get_role(assignment['role_id'])
        roles.append({'id': role['id'], 'name': role['name']})
    return roles