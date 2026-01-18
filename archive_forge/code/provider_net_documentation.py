from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import net
from heat.engine import support
Updates the resource with provided properties.

        Adds 'provider:' extension to the required properties during update.
        