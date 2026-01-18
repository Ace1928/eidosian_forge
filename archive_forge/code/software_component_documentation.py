from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine import support
from heat.rpc import api as rpc_api
Validate SoftwareComponent properties consistency.