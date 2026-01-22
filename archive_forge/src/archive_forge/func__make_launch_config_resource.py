from oslo_log import log as logging
from oslo_utils import excutils
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine.notification import autoscaling as notification
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import instance_group as instgrp
from heat.engine import rsrc_defn
from heat.engine import support
from heat.scaling import cooldown
from heat.scaling import scalingutil as sc_util
def _make_launch_config_resource(self, name, props):
    lc_res_type = 'AWS::AutoScaling::LaunchConfiguration'
    lc_res_def = rsrc_defn.ResourceDefinition(name, lc_res_type, props)
    lc_res = resource.Resource(name, lc_res_def, self.stack)
    return lc_res