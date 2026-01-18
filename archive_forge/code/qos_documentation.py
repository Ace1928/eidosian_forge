from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource for guaranteeing packet rate.

    This rule can be associated with a QoS policy, and then the policy
    can be used by a neutron port to provide guaranteed packet rate QoS
    capabilities.

    Depending on drivers the guarantee may be enforced on two levels.
    First when a server is placed (scheduled) on physical infrastructure
    and/or second in the data plane of the physical hypervisor. For details
    please see Neutron documentation:

    https://docs.openstack.org/neutron/latest/admin/config-qos-min-pps.html

    The default policy usage of this resource is limited to
    administrators only.
    