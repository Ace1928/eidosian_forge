from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class RemoveNetworkFromSpeaker(neutronv20.NeutronCommand):
    """Remove a network from the BGP speaker."""

    def get_parser(self, prog_name):
        parser = super(RemoveNetworkFromSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='BGP_SPEAKER', help=_('ID or name of the BGP speaker.'))
        parser.add_argument('network', metavar='NETWORK', help=_('ID or name of the network to remove.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _speaker_id = get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        _net_id = get_network_id(neutron_client, parsed_args.network)
        neutron_client.remove_network_from_bgp_speaker(_speaker_id, {'network_id': _net_id})
        print(_('Removed network %(net)s from BGP speaker %(speaker)s.') % {'net': parsed_args.network, 'speaker': parsed_args.bgp_speaker}, file=self.app.stdout)