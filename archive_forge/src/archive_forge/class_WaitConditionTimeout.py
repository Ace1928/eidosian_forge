import collections
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine.resources import signal_responder
class WaitConditionTimeout(exception.Error):

    def __init__(self, wait_condition, handle):
        reasons = handle.get_status_reason(handle.STATUS_SUCCESS)
        vals = {'len': len(reasons), 'count': wait_condition.properties[wait_condition.COUNT]}
        if reasons:
            vals['reasons'] = ';'.join(reasons)
            message = _('%(len)d of %(count)d received - %(reasons)s') % vals
        else:
            message = _('%(len)d of %(count)d received') % vals
        super(WaitConditionTimeout, self).__init__(message)