import collections
from heat.common.i18n import _
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
A resource for managing Neutron L2 Gateways.

    The are a number of use cases that can be addressed by an L2 Gateway API.
    Most notably in cloud computing environments, a typical use case is
    bridging the virtual with the physical. Translate this to Neutron and the
    OpenStack world, and this means relying on L2 Gateway capabilities to
    extend Neutron logical (overlay) networks into physical (provider)
    networks that are outside the OpenStack realm.
    