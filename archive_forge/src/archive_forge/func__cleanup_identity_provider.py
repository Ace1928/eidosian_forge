import uuid
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def _cleanup_identity_provider(self, service, resource_type, operation, payload):
    domain_id = payload['resource_info']
    hints = driver_hints.Hints()
    hints.add_filter('domain_id', domain_id)
    idps = self.driver.list_idps(hints=hints)
    for idp in idps:
        try:
            self.delete_idp(idp['id'])
        except exception.IdentityProviderNotFound:
            LOG.debug('Identity Provider %(idpid)s not found when deleting domain contents for %(domainid)s, continuing with cleanup.', {'idpid': idp['id'], 'domainid': domain_id})