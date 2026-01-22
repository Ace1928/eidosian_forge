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
def _cleanup_idp_domain(self, domain_id):
    domain = {'enabled': False}
    PROVIDERS.resource_api.update_domain(domain_id, domain)
    PROVIDERS.resource_api.delete_domain(domain_id)