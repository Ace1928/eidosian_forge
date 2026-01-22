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
def _create_derived_config(self, action, source_config):
    derived_params = self._build_derived_config_params(action, source_config)
    derived_config = self.rpc_client().create_software_config(self.context, **derived_params)
    return derived_config[rpc_api.SOFTWARE_CONFIG_ID]