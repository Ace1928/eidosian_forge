from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource for managing Neutron security groups.

    Security groups are sets of IP filter rules that are applied to an
    instance's networking. They are project specific, and project members can
    edit the default rules for their group and add new rules sets. All projects
    have a "default" security group, which is applied to instances that have no
    other security group defined.
    