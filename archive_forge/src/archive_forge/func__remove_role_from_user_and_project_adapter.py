import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
@notifications.role_assignment('deleted')
def _remove_role_from_user_and_project_adapter(self, role_id, user_id=None, group_id=None, domain_id=None, project_id=None, inherited_to_projects=False, context=None):
    self.driver.remove_role_from_user_and_project(user_id, project_id, role_id)
    payload = {'user_id': user_id, 'project_id': project_id}
    notifications.Audit.internal(notifications.REMOVE_APP_CREDS_FOR_USER, payload)
    self._invalidate_token_cache(role_id, group_id, user_id, project_id, domain_id)