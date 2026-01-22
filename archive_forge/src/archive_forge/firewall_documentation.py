from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource for the FirewallRule resource in Neutron FWaaS.

    FirewallRule represents a collection of attributes like ports,
    ip addresses etc. which define match criteria and action (allow, or deny)
    that needs to be taken on the matched data traffic.
    