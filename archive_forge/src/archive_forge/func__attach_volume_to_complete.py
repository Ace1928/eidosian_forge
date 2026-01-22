from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import volume_base as vb
from heat.engine import support
from heat.engine import translation
def _attach_volume_to_complete(self, prg_attach):
    if not prg_attach.called:
        prg_attach.called = self.client_plugin('nova').attach_volume(prg_attach.srv_id, prg_attach.vol_id, prg_attach.device)
        return False
    if not prg_attach.complete:
        prg_attach.complete = self.client_plugin().check_attach_volume_complete(prg_attach.vol_id)
        return prg_attach.complete