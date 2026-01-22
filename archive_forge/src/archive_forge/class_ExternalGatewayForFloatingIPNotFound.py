from neutron_lib._i18n import _
from neutron_lib import exceptions
class ExternalGatewayForFloatingIPNotFound(exceptions.NotFound):
    message = _('External network %(external_network_id)s is not reachable from subnet %(subnet_id)s.  Therefore, cannot associate Port %(port_id)s with a Floating IP.')