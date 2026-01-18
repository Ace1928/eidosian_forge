from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def trust_project_domain(self):
    if not self.__trust_project_domain:
        if self.trust:
            self.__trust_project_domain = PROVIDERS.resource_api.get_domain(self.trust_project['domain_id'])
    return self.__trust_project_domain