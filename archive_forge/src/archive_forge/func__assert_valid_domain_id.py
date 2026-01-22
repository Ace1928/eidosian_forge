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
def _assert_valid_domain_id(self, domain_id):
    PROVIDERS.resource_api.get_domain(domain_id)