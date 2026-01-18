from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine import support
from heat.engine import translation
A resource for IPsec policy in Neutron.

    The IP security policy specifying the authentication and encryption
    algorithm, and encapsulation mode used for the established VPN connection.
    