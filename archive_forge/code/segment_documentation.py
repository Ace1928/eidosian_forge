from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
A resource for Neutron Segment.

    This requires enabling the segments service plug-in by appending
    'segments' to the list of service_plugins in the neutron.conf.

    The default policy usage of this resource is limited to
    administrators only.
    