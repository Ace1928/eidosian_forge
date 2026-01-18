from oslo_config import cfg
from oslo_log import log as logging
import webob.exc
from glance.api import policy
from glance.common import exception
from glance.i18n import _
def update_property(self, name, value=None):
    if name == 'visibility':
        self._enforce_visibility(value)
    self.modify_image()