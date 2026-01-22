import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_manager_role(self):
    role = self._ensure_role_exists(self.manager_role_name)
    self.manager_role_id = role['id']
    self._ensure_implied_role(self.manager_role_id, self.member_role_id)