import uuid
from oslo_log import log
from keystone.common import driver_hints
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.server import backends
def _bootstrap_service_role(self):
    role = self._ensure_role_exists(self.service_role_name)
    self.service_role_id = role['id']