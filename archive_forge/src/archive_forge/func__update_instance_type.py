import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _update_instance_type(self, prop_diff):
    flavor = prop_diff[self.INSTANCE_TYPE]
    flavor_id = self.client_plugin().find_flavor_by_name_or_id(flavor)
    handler_args = {'args': (flavor_id,)}
    checker_args = {'args': (flavor_id,)}
    prg_resize = progress.ServerUpdateProgress(self.resource_id, 'resize', handler_extra=handler_args, checker_extra=checker_args)
    prg_verify = progress.ServerUpdateProgress(self.resource_id, 'verify_resize')
    return (prg_resize, prg_verify)