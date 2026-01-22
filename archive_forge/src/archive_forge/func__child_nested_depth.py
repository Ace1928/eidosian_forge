import json
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import resource
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
def _child_nested_depth(self):
    if self.stack.nested_depth >= cfg.CONF.max_nested_stack_depth:
        msg = _('Recursion depth exceeds %d.') % cfg.CONF.max_nested_stack_depth
        raise exception.RequestLimitExceeded(message=msg)
    return self.stack.nested_depth + 1