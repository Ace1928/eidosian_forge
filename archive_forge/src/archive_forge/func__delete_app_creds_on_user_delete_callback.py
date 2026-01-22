from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _delete_app_creds_on_user_delete_callback(self, service, resource_type, operation, payload):
    user_id = payload['resource_info']
    self._delete_application_credentials_for_user(user_id)
    self._delete_access_rules_for_user(user_id)