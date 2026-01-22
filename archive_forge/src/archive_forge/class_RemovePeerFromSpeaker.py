from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.common import validators
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.bgp import peer as bgp_peer
class RemovePeerFromSpeaker(neutronv20.NeutronCommand):
    """Remove a peer from the BGP speaker."""

    def get_parser(self, prog_name):
        parser = super(RemovePeerFromSpeaker, self).get_parser(prog_name)
        parser.add_argument('bgp_speaker', metavar='BGP_SPEAKER', help=_('ID or name of the BGP speaker.'))
        parser.add_argument('bgp_peer', metavar='BGP_PEER', help=_('ID or name of the BGP peer to remove.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _speaker_id = get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        _peer_id = bgp_peer.get_bgp_peer_id(neutron_client, parsed_args.bgp_peer)
        neutron_client.remove_peer_from_bgp_speaker(_speaker_id, {'bgp_peer_id': _peer_id})
        print(_('Removed BGP peer %(peer)s from BGP speaker %(speaker)s.') % {'peer': parsed_args.bgp_peer, 'speaker': parsed_args.bgp_speaker}, file=self.app.stdout)