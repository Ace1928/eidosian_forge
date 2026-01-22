from neutron_lib._i18n import _
from neutron_lib import exceptions
class SubnetIsNotConnectedToRouter(exceptions.BadRequest):
    message = _('Subnet %(subnet_id)s is not connected to Router %(router_id)s')