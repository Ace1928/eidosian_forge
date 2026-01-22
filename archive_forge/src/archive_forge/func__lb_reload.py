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
def _lb_reload(self, exclude=frozenset(), refresh_data=True):
    lb_names = self.properties.get(self.LOAD_BALANCER_NAMES) or []
    if lb_names:
        if refresh_data:
            self._outputs = None
        try:
            all_refids = self.get_output(self.OUTPUT_MEMBER_IDS)
        except (exception.NotFound, exception.TemplateOutputError) as op_err:
            LOG.debug('Falling back to grouputils due to %s', op_err)
            if refresh_data:
                self._nested = None
            instances = grouputils.get_members(self)
            all_refids = {i.name: i.FnGetRefId() for i in instances}
            names = [i.name for i in instances]
        else:
            group_data = self._group_data(refresh=refresh_data)
            names = group_data.member_names(include_failed=False)
        id_list = [all_refids[n] for n in names if n not in exclude and n in all_refids]
        lbs = [self.stack[name] for name in lb_names]
        lbutils.reconfigure_loadbalancers(lbs, id_list)