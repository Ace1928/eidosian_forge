from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0.bgp import speaker as bgp_speaker
class AddBGPSpeakerToDRAgent(neutronV20.NeutronCommand):
    """Add a BGP speaker to a Dynamic Routing agent."""

    def get_parser(self, prog_name):
        parser = super(AddBGPSpeakerToDRAgent, self).get_parser(prog_name)
        add_common_args(parser)
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _speaker_id = bgp_speaker.get_bgp_speaker_id(neutron_client, parsed_args.bgp_speaker)
        neutron_client.add_bgp_speaker_to_dragent(parsed_args.dragent_id, {'bgp_speaker_id': _speaker_id})
        print(_('Associated BGP speaker %s to the Dynamic Routing agent.') % parsed_args.bgp_speaker, file=self.app.stdout)