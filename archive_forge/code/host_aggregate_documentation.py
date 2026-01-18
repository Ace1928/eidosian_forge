from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
A resource for further partition an availability zone with hosts.

    While availability zones are visible to users, host aggregates are only
    visible to administrators. Host aggregates started out as a way to use
    Xen hypervisor resource pools, but has been generalized to provide a
    mechanism to allow administrators to assign key-value pairs to groups of
    machines. Each node can have multiple aggregates, each aggregate can have
    multiple key-value pairs, and the same key-value pair can be assigned to
    multiple aggregate. This information can be used in the scheduler to
    enable advanced scheduling, to set up xen hypervisor resources pools or to
    define logical groups for migration.
    