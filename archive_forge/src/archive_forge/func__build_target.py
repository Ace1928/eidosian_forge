from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def _build_target(self):
    target = {'project_id': self._context.project_id}
    if self._image:
        target = policy.ImageTarget(self._image)
    return target