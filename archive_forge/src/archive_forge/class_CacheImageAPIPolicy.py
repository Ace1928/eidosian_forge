from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
class CacheImageAPIPolicy(APIPolicyBase):

    def __init__(self, context, image=None, policy_str=None, target=None, enforcer=None):
        self._context = context
        target = {}
        self._image = image
        if self._image:
            target = policy.ImageTarget(self._image)
        self._target = target
        self.enforcer = enforcer or policy.Enforcer()
        self.policy_str = policy_str
        super(CacheImageAPIPolicy, self).__init__(context, target, enforcer)

    def manage_image_cache(self):
        self._enforce(self.policy_str)