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
def _validate_nested_resources(self, templ):
    if cfg.CONF.max_resources_per_stack == -1:
        return
    total_resources = len(templ[templ.RESOURCES]) + self.stack.total_resources(self.root_stack_id)
    identity = self.nested_identifier()
    if identity is not None:
        existing = self.rpc_client().list_stack_resources(self.context, identity)
        total_resources -= len(existing)
    if total_resources > cfg.CONF.max_resources_per_stack:
        message = exception.StackResourceLimitExceeded.msg_fmt
        raise exception.RequestLimitExceeded(message=message)