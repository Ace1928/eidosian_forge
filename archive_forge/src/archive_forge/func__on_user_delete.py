from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _on_user_delete(self, service, resource_type, operation, payload):
    user_id = payload['resource_info']
    trusts = self.driver.list_trusts_for_trustee(user_id)
    trusts = trusts + self.driver.list_trusts_for_trustor(user_id)
    for trust in trusts:
        self.driver.delete_trust(trust['id'])