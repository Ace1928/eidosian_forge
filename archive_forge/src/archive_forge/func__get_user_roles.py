from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _get_user_roles(self, user_id, project_id):
    assignment_list = self.assignment_api.list_role_assignments(user_id=user_id, project_id=project_id, effective=True)
    return list(set([x['role_id'] for x in assignment_list]))