from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
A resource for neutron networking-sfc.

    This resource used to define the service function path by arranging
    networking-sfc port-pair-groups and set of flow classifiers, to specify
    the classified traffic flows to enter the chain.
    