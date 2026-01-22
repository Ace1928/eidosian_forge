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
def _stack_kwargs(self, user_params, child_template, adopt_data=None):
    defn = self.child_definition(child_template, user_params)
    parsed_template = defn.t
    if adopt_data is None:
        template_id = parsed_template.store(self.context)
        return {'template_id': template_id, 'template': None, 'params': None, 'files': None}
    else:
        return {'template': parsed_template.t, 'params': defn.env.user_env_as_dict(), 'files': parsed_template.files}