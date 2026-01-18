from heat.common import exception
from heat.common.i18n import _
from heat.common import netutils
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource that implements neutron subnet pool.

    This resource can be used to create a subnet pool with a large block
    of addresses and create subnets from it.
    