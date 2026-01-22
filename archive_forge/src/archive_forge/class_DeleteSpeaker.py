from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class DeleteSpeaker(neutronv20.DeleteCommand):
    """Delete a BGP speaker."""
    resource = 'bgp_speaker'