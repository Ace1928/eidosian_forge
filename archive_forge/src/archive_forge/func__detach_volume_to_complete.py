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
def _detach_volume_to_complete(self, prg_detach):
    if not prg_detach.called:
        prg_detach.called = self.client_plugin('nova').detach_volume(prg_detach.srv_id, prg_detach.attach_id)
        return False
    if not prg_detach.cinder_complete:
        prg_detach.cinder_complete = self.client_plugin().check_detach_volume_complete(prg_detach.vol_id, prg_detach.srv_id)
        return False
    if not prg_detach.nova_complete:
        prg_detach.nova_complete = self.client_plugin('nova').check_detach_volume_complete(prg_detach.srv_id, prg_detach.attach_id)
        return False