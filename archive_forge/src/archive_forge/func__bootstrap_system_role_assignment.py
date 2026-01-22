import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_system_role_assignment(self):
    try:
        PROVIDERS.assignment_api.create_system_grant_for_user(self.admin_user_id, self.admin_role_id)
        LOG.info('Granted role %(role)s on the system to user %(username)s.', {'role': self.admin_role_name, 'username': self.admin_username})
    except exception.Conflict:
        LOG.info('User %(username)s already has role %(role)s on the system.', {'username': self.admin_username, 'role': self.admin_role_name})