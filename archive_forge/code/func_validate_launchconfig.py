import functools
from oslo_log import log as logging
from heat.common import environment_format
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import short_id
from heat.common import timeutils as iso8601utils
from heat.engine import attributes
from heat.engine import environment
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.scaling import lbutils
from heat.scaling import rolling_update
from heat.scaling import template
def validate_launchconfig(self):
    conf_refid = self.properties.get(self.LAUNCH_CONFIGURATION_NAME)
    if conf_refid:
        conf = self.stack.resource_by_refid(conf_refid)
        if conf is None:
            raise ValueError(_('%(lc)s (%(ref)s) reference can not be found.') % dict(lc=self.LAUNCH_CONFIGURATION_NAME, ref=conf_refid))
        if self.name not in conf.required_by():
            raise ValueError(_('%(lc)s (%(ref)s) requires a reference to the configuration not just the name of the resource.') % dict(lc=self.LAUNCH_CONFIGURATION_NAME, ref=conf_refid))