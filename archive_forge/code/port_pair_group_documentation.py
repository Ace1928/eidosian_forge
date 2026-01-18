from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
Heat Template Resource for networking-sfc port-pair-group.

    Multiple port-pairs may be included in a port-pair-group to allow the
    specification of a set of functionally equivalent Service Functions that
    can be used for load distribution.
    