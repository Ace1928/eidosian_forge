from neutron_lib._i18n import _
from neutron_lib import exceptions
from neutron_lib.exceptions import agent as agent_exc
class NetworkHostedByDHCPAgent(exceptions.Conflict):
    message = _('The network %(network_id)s has been already hosted by the DHCP Agent %(agent_id)s.')