from oslo_log import log as logging
from oslo_serialization import jsonutils
import tempfile
from heat.common import auth_plugin
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.common import template_format
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
from heat.engine import template
def _needs_update(self, after, before, after_props, before_props, prev_resource, check_init_complete=True):
    if self.state == (self.CHECK, self.FAILED):
        raise resource.UpdateReplace(self)
    return True