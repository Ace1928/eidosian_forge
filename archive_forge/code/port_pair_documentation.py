from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
A resource for neutron networking-sfc port-pair.

    This plug-in requires networking-sfc>=1.0.0. So to enable this
    plug-in, install this library and restart the heat-engine.

    A Port Pair represents a service function instance. The ingress port and
    the egress port of the service function may be specified. If a service
    function has one bidirectional port, the ingress port has the same value
    as the egress port.
    