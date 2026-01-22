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
def _child_parsed_template(self, child_template, child_env):
    parsed_template = self._parse_child_template(child_template, child_env)
    if not hasattr(type(self), 'attributes_schema'):
        self.attributes = None
        self._outputs_to_attribs(parsed_template)
    return parsed_template