from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
def _get_project_roles(self):
    roles = []
    project_roles = PROVIDERS.assignment_api.get_roles_for_user_and_project(self.user_id, self.project_id)
    for role_id in project_roles:
        r = PROVIDERS.role_api.get_role(role_id)
        roles.append({'id': r['id'], 'name': r['name']})
    return roles