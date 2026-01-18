from oslo_log import log
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import reflection
from keystone.common import cache
from keystone.common import provider_api
from keystone import exception
from keystone.i18n import _
@property
def trustor(self):
    if not self.__trustor:
        if self.trust:
            self.__trustor = PROVIDERS.identity_api.get_user(self.trust['trustor_user_id'])
    return self.__trustor