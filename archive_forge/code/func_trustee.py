from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def trustee(self):
    if not self.__trustee:
        if self.trust:
            self.__trustee = PROVIDERS.identity_api.get_user(self.trust['trustee_user_id'])
    return self.__trustee