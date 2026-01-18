from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def trust(self):
    if not self.__trust:
        if self.trust_id:
            self.__trust = PROVIDERS.trust_api.get_trust(self.trust_id)
    return self.__trust