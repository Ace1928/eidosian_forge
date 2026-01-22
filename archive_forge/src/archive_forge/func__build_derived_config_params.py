import copy
import itertools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import output
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import resource_group
from heat.engine.resources import signal_responder
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
def _build_derived_config_params(self, action, source):
    derived_inputs = self._build_derived_inputs(action, source)
    derived_options = self._build_derived_options(action, source)
    derived_config = self._build_derived_config(action, source, derived_inputs, derived_options)
    derived_name = self.properties.get(self.NAME) or source.get(rpc_api.SOFTWARE_CONFIG_NAME)
    return {rpc_api.SOFTWARE_CONFIG_GROUP: source.get(rpc_api.SOFTWARE_CONFIG_GROUP) or 'Heat::Ungrouped', rpc_api.SOFTWARE_CONFIG_CONFIG: derived_config or self.empty_config(), rpc_api.SOFTWARE_CONFIG_OPTIONS: derived_options, rpc_api.SOFTWARE_CONFIG_INPUTS: [i.as_dict() for i in derived_inputs], rpc_api.SOFTWARE_CONFIG_OUTPUTS: [o.as_dict() for o in source[rpc_api.SOFTWARE_CONFIG_OUTPUTS]], rpc_api.SOFTWARE_CONFIG_NAME: derived_name or self.physical_resource_name()}