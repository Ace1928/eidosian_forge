import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
def _check_scaling_allowed(self, cooldown):
    metadata = self.metadata_get()
    if metadata.get('scaling_in_progress'):
        LOG.info('Can not perform scaling action: resource %s is already in scaling.', self.name)
        reason = _('due to scaling activity')
        raise resource.NoActionRequired(res_name=self.name, reason=reason)
    cooldown = self._sanitize_cooldown(cooldown)
    if all((k not in metadata for k in ('cooldown', 'cooldown_end'))):
        metadata.pop('scaling_in_progress', None)
        if metadata and cooldown != 0:
            last_adjust = next(iter(metadata.keys()))
            if not timeutils.is_older_than(last_adjust, cooldown):
                self._log_and_raise_no_action(cooldown)
    elif 'cooldown_end' in metadata:
        cooldown_end = next(iter(metadata['cooldown_end'].keys()))
        now = timeutils.utcnow().isoformat()
        if now < cooldown_end:
            self._log_and_raise_no_action(cooldown)
    elif cooldown != 0:
        last_adjust = next(iter(metadata['cooldown'].keys()))
        if not timeutils.is_older_than(last_adjust, cooldown):
            self._log_and_raise_no_action(cooldown)
    metadata['scaling_in_progress'] = True
    self.metadata_set(metadata)