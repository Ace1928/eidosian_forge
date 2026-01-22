import datetime
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from oslo_log import log as logging
from oslo_utils import timeutils
def _log_and_raise_no_action(self, cooldown):
    LOG.info('Can not perform scaling action: resource %(name)s is in cooldown (%(cooldown)s).', {'name': self.name, 'cooldown': cooldown})
    reason = _('due to cooldown, cooldown %s') % cooldown
    raise resource.NoActionRequired(res_name=self.name, reason=reason)