import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _attach_volumes(self, attachers):
    for attacher in attachers:
        if not attacher.called:
            self.client_plugin().attach_volume(attacher.srv_id, attacher.vol_id, attacher.device)
            attacher.called = True
            return False
    for attacher in attachers:
        if not attacher.complete:
            attacher.complete = self.client_plugin('cinder').check_attach_volume_complete(attacher.vol_id)
            break
    out = all((attacher.complete for attacher in attachers))
    return out