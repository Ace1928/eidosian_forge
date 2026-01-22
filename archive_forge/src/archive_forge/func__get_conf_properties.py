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
def _get_conf_properties(self):
    conf_refid = self.properties[self.LAUNCH_CONFIGURATION_NAME]
    conf = self.stack.resource_by_refid(conf_refid)
    c_props = conf.frozen_definition().properties(conf.properties_schema, conf.context)
    props = {k: v for k, v in c_props.items() if k in c_props.data}
    for key in [conf.BLOCK_DEVICE_MAPPINGS, conf.NOVA_SCHEDULER_HINTS]:
        if props.get(key) is not None:
            props[key] = [{k: v for k, v in prop.items() if k in c_props.data[key][idx]} for idx, prop in enumerate(props[key])]
    if 'InstanceId' in props:
        props = conf.rebuild_lc_properties(props['InstanceId'])
    props['Tags'] = self._tags()
    props.pop('InstanceId', None)
    return (conf, props)