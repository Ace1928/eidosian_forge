from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
def _validate_handle_resource(self, handle):
    if handle is not None and isinstance(handle, wc_base.BaseWaitConditionHandle):
        return
    LOG.debug('Got %r instead of wait condition handle', handle)
    hn = handle.name if handle else self.properties[self.HANDLE]
    msg = _('%s is not a valid wait condition handle.') % hn
    raise ValueError(msg)