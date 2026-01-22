import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
def _finished_scaling(self, cooldown, cooldown_reason, size_changed=True):
    metadata = self.metadata_get()
    if size_changed:
        cooldown = self._sanitize_cooldown(cooldown)
        cooldown_end = (timeutils.utcnow() + datetime.timedelta(seconds=cooldown)).isoformat()
        if 'cooldown_end' in metadata:
            cooldown_end = max(next(iter(metadata['cooldown_end'].keys())), cooldown_end)
        metadata['cooldown_end'] = {cooldown_end: cooldown_reason}
    metadata['scaling_in_progress'] = False
    try:
        self.metadata_set(metadata)
    except exception.NotFound:
        pass