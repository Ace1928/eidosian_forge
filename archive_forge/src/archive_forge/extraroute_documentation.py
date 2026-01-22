from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
Resource for specifying extra routes for Neutron router.

    Resource allows to specify nexthop IP and destination network for router.
    