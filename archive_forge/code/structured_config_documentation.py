import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
This resource associates a group of servers with some configuration.

    This resource works similar as OS::Heat::SoftwareDeploymentGroup, but for
    structured resources.
    