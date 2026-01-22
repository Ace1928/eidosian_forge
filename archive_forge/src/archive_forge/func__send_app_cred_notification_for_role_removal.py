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
def _send_app_cred_notification_for_role_removal(self, role_id):
    """Delete all application credential for a specific role.

        :param role_id: role identifier
        :type role_id: string
        """
    assignments = self.list_role_assignments(role_id=role_id)
    for assignment in assignments:
        if 'user_id' in assignment and 'project_id' in assignment:
            payload = {'user_id': assignment['user_id'], 'project_id': assignment['project_id']}
            notifications.Audit.internal(notifications.REMOVE_APP_CREDS_FOR_USER, payload)